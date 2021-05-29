import cv2
import numpy as np


# Facility for show in notebook
def show(imgs, scale=1, names=None):
    if names is None:
        names = map(str,range(len(imgs)))
    cv2.startWindowThread()
    for name, img in zip(names, imgs):
        cv2.imshow(name, resize(img, scale))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_color(img):
    return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)


def blur(img, ksize=5):
    return cv2.GaussianBlur(img,(ksize,ksize),0)


def resize(img, scale):
    dsize = (np.array(img.shape)[:2]*scale).astype(int)
    return cv2.resize(img, (dsize[1], dsize[0]), interpolation=cv2.INTER_AREA)


def to_bin(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


def close(img, s):
    kernel = np.ones((s,s),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def normalize(img):
    img = (img-img.min()) / (img.max() - img.min())
    return img*2 - 1