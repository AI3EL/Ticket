import pytesseract
import cv2
from img_utils import get_color, show
import matplotlib.pyplot as plt


def postprocess_ocr(df_ocr):
    df_ocr = rm_weird_height(df_ocr)
    df_ocr = rm_above_title(df_ocr)
    return df_ocr


def rm_weird_height(df_ocr, low_rtol=0.6, high_rtol = 2, v=True):
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
    out = []
    for line_id in sorted(df_ocr.line_num.unique()):
        df_line_words = df_ocr[(df_ocr.line_num == line_id) & (df_ocr.level == 5)]
        line_words = df_line_words.sort_values('left')['text']
        out.append(' '.join(line_words))
    return out


def show_boxes(img, scale=1):
    img = get_color(img)
    h = img.shape[0]
    boxes = pytesseract.image_to_boxes(img) 
    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
    show([img], scale=scale)


def print_ocr(img, custom_config='-l fra --oem 3 --psm 6'):
    print(get_ocr(img, custom_config))


def get_ocr(img, custom_config='-l fra --oem 3 --psm 6'):
    return pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME, config=custom_config)
