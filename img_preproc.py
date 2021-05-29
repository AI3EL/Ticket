from tes_ocr import TARGET_DPI
from img_utils import blur, get_color, get_grayscale, normalize, resize, show, to_bin
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.linear_model import RANSACRegressor
import imutils

TICKET_WIDTH_CM = 7.
TICKET_WIDTH_IN = TICKET_WIDTH_CM * 0.3937
CROP_LEFT = 90/300
CROP_RIGHT = 50/300
TOP_CROP_ADD = 50/300
BOTTOM_CROP_ADD = 50/300
DOT_SPACING = 17.75/300

def preprocess_ocr(img, v=0, target_dpi=TARGET_DPI):
    gray_img = get_grayscale(img)
    blur_img = blur(gray_img, 5)
    th, bin_img = to_bin(blur_img)
    contour = detect_ticket(bin_img, 0)
    h, status = get_hom_from_cont(img, contour)

    ocr_img = cv2.warpPerspective(img, h, (gray_img.shape[1],gray_img.shape[0]))
    dpi = get_dpi(ocr_img)
    if v:
        print(f'DPI: {dpi:.0f}')
    if v > 1:
        show([ocr_img, gray_img], scale=0.2)
    ocr_scale = target_dpi / dpi

    ocr_img = resize(ocr_img, ocr_scale)
    ocr_img = crop_width(ocr_img, target_dpi)
    return ocr_img


def detect_ticket(bin_img, v=0):
    contours = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:1] 
    if v:
        col_img = get_color(bin_img)
        cv2.drawContours(col_img, contours, -1, (0,255,0), 3)
        show([col_img], 0.2)
    return contours[0][:,0]

# labelling the lines as vertical or horizontal based on derivative was not great
# could we improve by using appx knowing quad ? 
def get_hom_from_cont(img, contour):
    h,w = img.shape[:2]

    peri = cv2.arcLength(contour, True)
    src_pts = cv2.approxPolyDP(contour, 0.02 * peri, True)[:,0]
    distance = src_pts[:,0]**2 + src_pts[:,1]**2
    i_roll = -distance.argmin()
    src_pts = np.roll(src_pts, i_roll, 0)
    distances = np.diff(src_pts, axis=0, append=src_pts[0:1])
    distances = (distances**2).sum(axis=1)

    # Landscape mode
    if distances[[0,2]].sum() > distances[[1,3]].sum():
        # TODO: up or down, other than best ocr results ... ?
        dst_pts = np.array([(0,0), (0,h), (w,h), (w,0)])
    else:
        dst_pts = np.array([(0,0), (0,h), (w,h), (w,0)])

    homography, status = cv2.findHomography(src_pts, dst_pts)

    return homography, status


# img should have the ticket width as width
def get_dpi(img):
    return img.shape[1]/TICKET_WIDTH_IN


def crop_width(img, dpi=TARGET_DPI):
    xmin = int(CROP_LEFT * dpi)
    xmax = img.shape[1] - int(CROP_RIGHT * dpi)
    return img[:, xmin:xmax]


def crop_height(gray_img, dpi = TARGET_DPI, th=120, min_black=20):
    ymin, ymax = 0, gray_img.shape[0]
    hsum = gray_img.mean(axis=1)
    is_black_line = hsum < th
    is_white_line = hsum > th
    if is_black_line[:min_black].all():
        ymin = is_white_line.argmax() + int(TOP_CROP_ADD * dpi/TARGET_DPI)
    if is_black_line[ymax-min_black].all():
        ymax -= np.flip(is_white_line).argmax() + int(BOTTOM_CROP_ADD * dpi/TARGET_DPI)
    return gray_img[ymin:ymax]


def detect_trailing_dots(img, dot_space=DOT_SPACING, n_space=6, score_th=25):
    img = normalize(img)

    h,w = img.shape
    crop = 15
    
    kernel = np.ones((11,17)) / (11*17-7*5) * 5
    kernel[3:8,5:12] = 0 
    kernel[4:7, 6:11] = -1 / (3*5)
    
    line_scores = []
    for shift in range(1, int(dot_space)+1):
        xs = w- shift - crop - (np.arange(n_space)*dot_space).astype(int)
        convolved = cv2.filter2D(img,-1,kernel)
        line_scores.append(convolved[:, xs].sum(axis=1))

    line_scores = np.array(line_scores)
    scores = line_scores.max(axis=0)
    shifts = line_scores.argmax(axis=0)

    plt.plot(scores)
    plt.show()

    plt.plot(line_scores.argmax(axis=0))
    plt.show()

    return np.array([(y, w-x) for y, (x, score) in enumerate(zip(shifts, scores)) if score > score_th])




# OLD

# Computes homography based on ticket borders
# Fit a ransac on left and right borders direction independently 
# as otherwise, ransac select only one side ...
# def get_ticket_homography(img, contours, residual_threshold=50, min_samples=0.5, v=0):
#     h,w = img.shape

#     xmins = np.array([np.argmax(img[y]) for y in range(h)])
#     xmaxs = np.array([w - np.argmax(np.flip(img[y])) for y in range(h)])

#     ys = np.arange(h).reshape(-1,1)
#     xmin_fit = RANSACRegressor(
#         min_samples=min_samples,
#         residual_threshold=residual_threshold,
#         ).fit(ys, xmins)

#     top_left_src = np.array([xmin_fit.predict([[0]])[0], 0])
#     bottom_left_src = np.array([xmin_fit.predict([[h]])[0], h])
    
#     if v:
#         inliers = xmin_fit.inlier_mask_
#         print('Min inlier prop', np.mean(inliers))
#         plt.plot(ys, xmin_fit.predict(ys), '--x')
#         plt.plot(ys, xmins, '--x')
#         plt.plot(ys[inliers], xmins[inliers], '--x')
#         plt.show()

#     xmax_fit = RANSACRegressor(
#         min_samples=min_samples, 
#         residual_threshold=residual_threshold,
#     ).fit(ys, xmaxs)
#     top_right_src = np.array([xmax_fit.predict([[0]])[0], 0])
#     bottom_right_src = np.array([xmax_fit.predict([[h]])[0], h])

#     if v:
#         inliers = xmax_fit.inlier_mask_
#         print('Max inlier prop', np.mean(inliers))
#         plt.plot(ys, xmax_fit.predict(ys), '--x')
#         plt.plot(ys, xmaxs, '--x')
#         plt.plot(ys[inliers], xmaxs[inliers], '--x')
#         plt.show()

#     if v > 1:
#         col_img = get_color(img)
#         col_img = cv2.line(col_img, top_left_src.astype(int), bottom_left_src.astype(int), (0,255,0), 5)
#         col_img = cv2.line(col_img, top_right_src.astype(int), bottom_right_src.astype(int), (0,255,0), 5)
#         show([col_img], 0.2)

#     src_pts = np.array([top_left_src, top_right_src, bottom_left_src, bottom_right_src])
#     dst_pts = np.array([(0,0), (w,0), (0,h), (w,h)])

#     homography, status = cv2.findHomography(src_pts, dst_pts)

#     return homography, status


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



# Also :
# Select local maxima according to robust criterion
# classify according to max value
# Neeed x upperbound

# Good example that it cannot work for Image 1.b 
# small hole that makes small bump before T bump, 
# better alternative to detect the Ts with tesseract or other 
# detection
# def get_T_bounds2(img, n_roll = 50, n_consec=30, v=False):
#     vmass = (255-img).sum(axis=0)
#     vmass = pd.Series(vmass).rolling(n_roll, center=True).mean()

#     vmass_increase = (vmass.diff()>0).rolling(n_consec, center=True).min()
#     vmass_decrease = (vmass.diff()<0).rolling(n_consec, center=True).min()
#     first_inc = vmass_increase.argmax()
#     dec_after_inc = first_inc + vmass_decrease[first_inc:].argmax()
#     second_inc = dec_after_inc + vmass_increase[dec_after_inc:].argmax()

    
#     vmass[:500].plot()
#     plt.axvline(first_inc, linestyle='--', color='b', label='first_inc')
#     plt.axvline(dec_after_inc, linestyle='--', color='r', label='dec_after_inc')
#     plt.axvline(second_inc, linestyle='--', color='g', label='sec_inc')
#     plt.show()

#     return first_inc, second_inc

# def pca_rotate(bin_img, v=0):
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
#     return center, theta