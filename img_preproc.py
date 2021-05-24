from os import TMP_MAX
from img_utils import get_color, show
import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import RANSACRegressor

# Also :
# Select local maxima according to robust criterion
# classify according to max value
# Neeed x upperbound

# Good example that it cannot work for Image 1.b 
# small hole that makes small bump before T bump, 
# better alternative to detect the Ts with tesseract or other 
# detection
def get_T_bounds2(img, n_roll = 50, n_consec=30, v=False):
    vmass = (255-img).sum(axis=0)
    vmass = pd.Series(vmass).rolling(n_roll, center=True).mean()

    vmass_increase = (vmass.diff()>0).rolling(n_consec, center=True).min()
    vmass_decrease = (vmass.diff()<0).rolling(n_consec, center=True).min()
    first_inc = vmass_increase.argmax()
    dec_after_inc = first_inc + vmass_decrease[first_inc:].argmax()
    second_inc = dec_after_inc + vmass_increase[dec_after_inc:].argmax()

    
    vmass[:500].plot()
    plt.axvline(first_inc, linestyle='--', color='b', label='first_inc')
    plt.axvline(dec_after_inc, linestyle='--', color='r', label='dec_after_inc')
    plt.axvline(second_inc, linestyle='--', color='g', label='sec_inc')
    plt.show()

    return first_inc, second_inc


# Computes homography based on ticket borders
# Fit a ransac on left and right borders direction independently 
# as otherwise, ransac select only one side ...
# debug info is commented
def get_ticket_homography(img, residual_threshold=20, min_samples=0.5):
    h,w = img.shape

    xmins = np.array([np.argmax(img[y]) for y in range(h)])
    xmaxs = np.array([w - np.argmax(np.flip(img[y])) for y in range(h)])

    ys = np.arange(h).reshape(-1,1)
    xmin_fit = RANSACRegressor(
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        ).fit(ys, xmins)

    top_left_src = np.array([xmin_fit.predict([[0]])[0], 0])
    bottom_left_src = np.array([xmin_fit.predict([[h]])[0], h])
    
    # inliers = xmin_fit.inlier_mask_
    # print('Min inlier prop', np.mean(inliers))
    # plt.plot(ys, xmin_fit.predict(ys), '--x')
    # plt.plot(ys, xmins, '--x')
    # plt.plot(ys[inliers], xmins[inliers], '--x')
    # plt.show()

    xmax_fit = RANSACRegressor(
        min_samples=min_samples, 
        residual_threshold=residual_threshold,
    ).fit(ys, xmaxs)
    top_right_src = np.array([xmax_fit.predict([[0]])[0], 0])
    bottom_right_src = np.array([xmax_fit.predict([[h]])[0], h])

    # inliers = xmax_fit.inlier_mask_
    # print('Max inlier prop', np.mean(inliers))
    # plt.plot(ys, xmax_fit.predict(ys), '--x')
    # plt.plot(ys, xmaxs, '--x')
    # plt.plot(ys[inliers], xmaxs[inliers], '--x')
    # plt.show()

    # col_img = get_color(img)
    # col_img = cv2.line(col_img, top_left_src.astype(int), bottom_left_src.astype(int), (0,255,0), 5)
    # col_img = cv2.line(col_img, top_right_src.astype(int), bottom_right_src.astype(int), (0,255,0), 5)
    # show([col_img], 0.2)

    src_pts = np.array([top_left_src, top_right_src, bottom_left_src, bottom_right_src])
    dst_pts = np.array([(0,0), (w,0), (0,h), (w,h)])

    homography, status = cv2.findHomography(src_pts, dst_pts)

    return homography, status


# OLD


# def get_homography(img, src_pts):
#     dst_pts = get_dst_pts(src_pts)
#     src_pts, dst_pts = src_pts.astype(int), dst_pts.astype(int)
#     return cv2.findHomography(src_pts, dst_pts)

# def get_dst_pts(src_pts):
#     a = src_pts[0]
#     b = (src_pts[1][0], a[1])
#     c = (a[0], src_pts[2][1])
#     d = (b[0], c[1])
#     return np.array([a,b,c,d])

# def pca_rotate(bin_img):
#     height, width = bin_img.shape
#     center = (height-1)/2,(width-1)/2
#     out = []
#     for i in tqdm(range(height)):
#         for j in range(width):
#             if bin_img[i,j]:
#                 out.append((i-center[0],j-center[1]))
#     img_pos = np.array(out, dtype=int)
#     print('Fitting PCA')
#     pca_fit = PCA(2).fit(img_pos)
#     print('PCA fitted')
#     x_h, x_w = pca_fit.components_ # Should have unit L2 norm
#     if np.abs(x_h[0]) < np.abs(x_w[0]):
#         x_h, x_w = x_w, x_h
#     if x_h[0] < 0:
#         x_h *= -1
#     theta = np.arccos(x_h[0])
#     print('Theta: ', theta)
#     rot = cv2.getRotationMatrix2D(center, theta, 1)
#     return cv2.warpAffine(bin_img, rot,(width, height))

