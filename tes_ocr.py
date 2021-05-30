from img_preproc import detect_trailing_dots
import pytesseract
import cv2
from img_utils import get_color, resize, show
import matplotlib.pyplot as plt
import numpy as np

BASE_CONFIG = '-l fra --oem 3 --psm 6'
# CONFIG = BASE_CONFIG + f' --dpi {TARGET_DPI:.0f}' + ' configs/abel_config1'
# CONFIG = BASE_CONFIG + f' --dpi {TARGET_DPI:.0f}' + ' /usr/share/tesseract-ocr/4.00/tessdata/configs/config_abel_1'
CONFIG = BASE_CONFIG + ' --dpi {:.0f} abel_config_1'


# Ideas:
# - remove trailing dots per line
# - detect price on right and allow digits only / , / € using "monoprix.patterns"
# - tesseract seems to better assess word sizes (apart when there are trailing dots)
# - detected height is not very precise
# Issue: slight skew --> vpad, but still has issue on line detection
TOTAL_PRICE_LEFT_IN = 1900 /1097
TOTAL_PRICE_RIGHT_IN = 2350 / 1097

UNIT_PRICE_LEFT_IN = 375 /300
UNIT_PRICE_RIGHT_IN = 480 / 300

DISC_PRICE_LEFT_IN = 470 /300
DISC_PRICE_RIGHT_IN = 590 / 300

# TODO: treat title sections and othes
# TODO: apply closing before or after binarization
# TODO: treat category trailing dots 
class MonoprixLineOCR:

    def __init__(self, ocr_dpi=300, v=0) -> None:
        self.v = v
        self.ocr_dpi = ocr_dpi
        self.line_config = f'-l fra --oem 3 --psm 7  --dpi {self.ocr_dpi}'
        self.article_config = f'-l fra --oem 3 --psm 7  --dpi {self.ocr_dpi} abel_article_config'
        self.price_config = f'--oem 3 --psm 8  --dpi {self.ocr_dpi} abel_pos_price_config'
        self.disc_config = f'--oem 3 --psm 8  --dpi {self.ocr_dpi} abel_neg_price_config'
        
    def fit(self, gray_img, bin_img, dpi):
        self.dpi = dpi
        vpad = int(2.6 *dpi / 300)

        line_starts, line_ends = self.find_lines(bin_img)

        line_ocrs = []
        for beg, end in zip(line_starts, line_ends):
            line_gray_img = gray_img[beg-vpad:end+vpad]
            line_bin_img = bin_img[beg-vpad:end+vpad]
            df_ocr = self.get_ocr(line_gray_img, self.line_config)
            df_words = df_ocr[df_ocr.level == 5]
            line_type, line_bin_img = self.assign_line_type(df_words, line_bin_img)

            print(line_type)

            line_gray_img = bin_img[beg-vpad:end+vpad]

            if line_type == 'Article':
                line_ocrs.append(self.read_article_line(line_bin_img, df_words))
            if line_type == 'Discount':
                line_ocrs.append(self.read_discount_line(line_bin_img, df_words))
            if line_type == 'Category':
                line_ocrs.append(self.read_category_line(line_bin_img))
            
            print(line_ocrs[-1])
        return line_ocrs

    def find_lines(self, bin_img):
        th = 5e3 * self.dpi / 300
        hmass = (255-bin_img).sum(axis=1)

        if self.v:
            plt.plot(hmass)
            plt.axhline(th, linestyle='--', color='k')
            plt.show()

        is_text = hmass > th
        line_bounds = np.diff(is_text.astype(int))
        line_starts = np.where(line_bounds == 1)[0]
        line_ends = np.where(line_bounds == -1)[0]
        if len(line_starts) > len(line_ends):
            line_ends = np.r_[line_ends, len(is_text)-1]
        return line_starts, line_ends

    def get_box_in(self, df_words, left, right):
        left, right = left * self.dpi, right * self.dpi
        word_left = df_words['left']
        word_right = word_left + df_words['width']
        mask = word_left > left
        mask &= word_right < right
        if any(mask):
            return (word_left[mask].iloc[0], word_right.iloc[1])
        return None

    def assign_line_type(self, df_words, line_bin_img):
        has_dots, line_bin_img = detect_trailing_dots(line_bin_img, self.dpi)
        if has_dots:
            print('Has dots !')
            return 'Category', line_bin_img

        if self.get_box_in(df_words, TOTAL_PRICE_LEFT_IN, TOTAL_PRICE_RIGHT_IN) is not None:
            return 'Article', line_bin_img
        
        if self.get_box_in(df_words, DISC_PRICE_LEFT_IN, DISC_PRICE_RIGHT_IN) is not None:
            return 'Discount', line_bin_img
        
        return 'Category', line_bin_img

    def get_ocr(self, img, config):
        scaled_img = resize(img, self.ocr_dpi/self.dpi)
        df_ocr = pytesseract.image_to_data(scaled_img, output_type=pytesseract.Output.DATAFRAME, config=config)
        df_ocr[['left', 'width', 'top', 'height']] *= self.dpi/self.ocr_dpi
        if self.v > 1:
            show_boxes(img, df_ocr, scale=1)
        return df_ocr

    def read_article_line(self, line_img, df_words):
        out = {}

        unit_price_box = self.get_box_in(df_words, UNIT_PRICE_LEFT_IN, UNIT_PRICE_RIGHT_IN)

        # Treat unit price
        if unit_price_box is not None:
            print('Treat unit price')
            right_article = int(UNIT_PRICE_LEFT_IN*self.dpi) - 5
            price_left, price_right = int(UNIT_PRICE_LEFT_IN*self.dpi), int(UNIT_PRICE_RIGHT_IN*self.dpi)
            price_img = line_img[:, price_left:price_right]
            df_ocr = self.get_ocr(price_img, self.price_config)
            out['unit_price'] = df_ocr[df_ocr.level == 5].text.values[0]
        else:
            right_article = int(TOTAL_PRICE_LEFT_IN*self.dpi) - 5

        # Treat article
        print('Treat article')
        article_img = line_img[:,:right_article]
        df_ocr = self.get_ocr(article_img, self.article_config)
        out['article_text'] = ' '.join(df_ocr[df_ocr.level == 5].text.values)

        # Treat Total Price
        print('Treat total price')
        price_left, price_right = int(TOTAL_PRICE_LEFT_IN*self.dpi), int(TOTAL_PRICE_RIGHT_IN*self.dpi)
        price_img = line_img[:,price_left:price_right]
        df_ocr = self.get_ocr(price_img, self.price_config)
        out['total_price'] = df_ocr[df_ocr.level == 5].text.values[0]

        return out
            

    def read_discount_line(self, line_img, df_words):
        out = {}

        # Treat disc price
        price_left, price_right = int(DISC_PRICE_LEFT_IN*self.dpi), int(DISC_PRICE_RIGHT_IN*self.dpi)
        price_img = line_img[:, price_left:price_right]
        df_ocr = self.get_ocr(price_img, self.disc_config)
        out['unit_price'] = df_ocr[df_ocr.level == 5].text.values[0]

        right_disc_text = int(DISC_PRICE_LEFT_IN*self.dpi) - 5

        # Treat disc_text
        disc_text_img = line_img[:,:right_disc_text]
        df_ocr = self.get_ocr(disc_text_img, self.article_config)
        out['disc_text'] = ' '.join(df_ocr[df_ocr.level == 5].text.values)

        return out    

    # TODO improve
    def read_category_line(self, line_img):
        out = {}
        df_ocr = self.get_ocr(line_img, self.article_config)
        out['category'] = ' '.join(df_ocr[df_ocr.level == 5].text.values)
        return out         


def postprocess_ocr(df_ocr):
    df_ocr = rm_weird_height(df_ocr)
    # df_ocr = rm_above_title(df_ocr)
    return df_ocr


def rm_weird_height(df_ocr, low_rtol=0.6, high_rtol = 2, v=0):
    median_word_height = df_ocr[df_ocr.level == 5].height.median()
    if v:
        mask = df_ocr.level == 5
        mask &= df_ocr.height > median_word_height*low_rtol/2
        mask &= df_ocr.height < median_word_height*high_rtol*1.1
        df_ocr[mask].height.plot.hist(bins=30)
        plt.axvline(median_word_height*low_rtol, linestyle='--', color='k')
        plt.axvline(median_word_height*high_rtol, linestyle='--', color='k')
        plt.show()
    mask = df_ocr.height > median_word_height*low_rtol
    mask &= df_ocr.height < median_word_height*high_rtol
    mask |= df_ocr.level != 5
    return df_ocr[mask]


def rm_above_title(df_ocr):
    title_h = df_ocr[df_ocr['text'] == 'MONOPRIX']['top']
    assert len(title_h) == 1
    title_h = title_h.iloc[0]
    mask = df_ocr['top'] >= title_h
    mask |= df_ocr.level != 5
    return df_ocr[mask]


def extract_text(df_ocr):
    df_words = df_ocr[df_ocr.level==5].copy()
    ids = ['page_num', 'par_num', 'block_num', 'line_num']
    df_words['line_id'] = df_words.apply(lambda row: tuple(row[id] for id in ids), axis=1) 
    df_words = df_words.sort_values(by=ids)
    line_words = df_words.groupby('line_id').apply(lambda df:' '.join(df['text']))
    return line_words.values


def show_boxes(img, df_ocr, level=5, scale=0.8):
    h, w = img.shape
    col_img = get_color(img)
    for _,row in df_ocr[df_ocr.level == level].iterrows():
        beg = (int(row['left']), int(row['top']))
        end = (min(int(row['left'] + row['width']), w-1), min(int(row['top']+row['height']), h-1))
        col_img = cv2.rectangle(col_img, beg, end, (0, 255, 0), 2)
    show([col_img], scale=scale)


def print_ocr(img, dpi, config=CONFIG):
    print(get_ocr(img, dpi, config))


def get_ocr(img, dpi, config=CONFIG):
    config = config.format(dpi)
    return pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME, config=config)
