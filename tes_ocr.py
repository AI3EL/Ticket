import pytesseract
import cv2
from img_utils import get_color, show
import matplotlib.pyplot as plt

BASE_CONFIG = '-l fra --oem 3 --psm 6'
TARGET_DPI = 1000
CONFIG = BASE_CONFIG + f' --dpi {TARGET_DPI:.0f}'


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
    col_img = get_color(img)
    for _,row in df_ocr[df_ocr.level == level].iterrows():
        beg = (row['left'], row['top'])
        end = (row['left'] + row['width'], row['top']+row['height'])
        col_img = cv2.rectangle(col_img, beg, end, (0, 255, 0), 2)
    show([col_img], scale=scale)


def print_ocr(img, custom_config=CONFIG):
    print(get_ocr(img, custom_config))


def get_ocr(img, custom_config=CONFIG):
    return pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME, config=custom_config)
