from os import stat
from img_utils import blur, denormalize, dilate, get_color, get_grayscale, normalize, resize, show, to_bin
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.linear_model import RANSACRegressor
import imutils

TICKET_WIDTH_CM = 7.
TICKET_WIDTH_IN = TICKET_WIDTH_CM * 0.3937
CROP_LEFT_IN = 90/300
CROP_RIGHT_IN = 50/300
TOP_CROP_ADD_IN = 50/300
BOTTOM_CROP_ADD_IN = 50/300
DOT_SPACING_IN = 17.75/300

def preprocess_ocr(img, target_dpi=None, v=0):
    gray_img = get_grayscale(img)
    blur_img = blur(gray_img, 5)
    th, bin_img = to_bin(blur_img)
    contour = detect_ticket(bin_img, v)
    h, status = get_hom_from_cont(img, contour)

    ocr_img = cv2.warpPerspective(img, h, (gray_img.shape[1],gray_img.shape[0]))
    dpi = get_dpi(ocr_img)
    if target_dpi is None:
        target_dpi = dpi
    if v:
        print(f'DPI: {dpi:.0f}')
    if v > 1:
        show([ocr_img, gray_img], scale=0.2)
    ocr_scale = target_dpi / dpi

    ocr_img = resize(ocr_img, ocr_scale)
    ocr_img = crop_width(ocr_img, target_dpi)
    return ocr_img, target_dpi


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

    edge_len = np.diff(src_pts, axis=0, append=src_pts[0:1])
    edge_len = (edge_len**2).sum(axis=1)

    dst_pts = np.array([(0,0), (0,h), (w,h), (w,0)])

    # Landscape mode TODO: find up and down
    if edge_len[[0,2]].sum() < edge_len[[1,3]].sum():
        dst_pts = np.roll(dst_pts, -1, 0)

    homography, status = cv2.findHomography(src_pts, dst_pts)

    return homography, status


# Finds the dpi of the img, knowing the ticket width
# img should have the ticket width as width
def get_dpi(img):
    return img.shape[1]/TICKET_WIDTH_IN


# Removes the T (Monoprix specific)
def crop_width(img, dpi):
    xmin = int(CROP_LEFT_IN * dpi)
    xmax = img.shape[1] - int(CROP_RIGHT_IN * dpi)
    return img[:, xmin:xmax]


def detect_trailing_dots(img, dpi, target_dpi=300, dot_space=None, n_space=6, point_th=5.0,  line_th= 25.0, v=0):
    assert target_dpi == 300

    # Resize for performances
    scaled_img = resize(img, target_dpi/dpi)

    if dot_space is None:
        dot_space = DOT_SPACING_IN * target_dpi

    h,w = scaled_img.shape
    crop = 15
    
    kernel = np.ones((11,17)) / (11*17-7*5) * 5
    kernel[3:8,5:12] = 0 
    kernel[4:7, 6:11] = -1 / (3*5)
    
    line_scores = []
    convolved = cv2.filter2D(normalize(scaled_img),-1,kernel)
    for shift in range(1, int(dot_space)+1):
        xs = w- shift - crop - (np.arange(n_space)*dot_space).astype(int)
        line_scores.append(convolved[:, xs].sum(axis=1))

    line_scores = np.array(line_scores)
    scores = line_scores.max(axis=0)
    shifts = line_scores.argmax(axis=0)

    if v>1:
        plt.plot(scores)
        plt.show()

        plt.plot(line_scores.argmax(axis=0))
        plt.show()

    y_dot_lines = [y for y, (_, score) in enumerate(zip(shifts, scores)) if score > line_th]
    
    # If no dot detected:
    if not len(y_dot_lines):
        return [], img


    print('Dots, detected, max score', scores.max())

    # Select center of dots that are in the lines detected as dot lines
    to_rm = ((convolved > point_th)*255).astype(np.uint8)
    not_in_start_dots = ~np.isin(np.arange(convolved.shape[0]), y_dot_lines)
    to_rm[not_in_start_dots] = 0

    # Resize to original shape
    dsize = (np.array(img.shape)[:2]).astype(int)
    to_rm = cv2.resize(to_rm, (dsize[1], dsize[0]), interpolation=cv2.INTER_AREA)

    # Get the connected components of the targets to remove
    _, comps, stats, centroids = cv2.connectedComponentsWithStats(255-img)
    comps_to_rm = np.unique(np.where(to_rm>127, comps, -1))[2:]  # Should remove -1 and 0
    px_to_rm = (np.isin(comps, comps_to_rm)*255).astype(np.uint8)
    px_to_rm = dilate(px_to_rm, 5)


    px_to_rm = (255-img)*px_to_rm*255
    out_img = 255-((255-img) - px_to_rm)

    if v:
        show([img, out_img, px_to_rm], 0.2)

    # Give one y point per dot removed [1,2,3,4,7,8] --> [2,7]
    prec = y_dot_lines[0]
    current_group = [prec]
    y_dot_centers = []
    for y in y_dot_lines[1:]:
        if y > prec+1:
            center = current_group[len(current_group)//2]
            center = int(center*dpi/target_dpi)
            y_dot_centers.append(center)
            current_group = [y]
        else:
            current_group.append(y)
        prec = y
    center = current_group[len(current_group)//2]
    center = int(center*dpi/target_dpi)
    y_dot_centers.append(center)    
    return y_dot_centers, out_img


# img should be Black on White
# Ref: https://www.hpl.hp.com/techreports/94/HPL-94-113.pdf
def get_skewed_lines(img, dpi, alpha=0.6, min_height=25/1097, v=0):
    img = 255-img
    min_height *= dpi

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img) # Might be reused afterward for optim
    left_order = np.argsort(stats[:, cv2.CC_STAT_LEFT])

    # Select only blobs(=letters) that have minimum height
    sel_blobs = [blob_id for blob_id in range(1, n_labels) if stats[blob_id, cv2.CC_STAT_HEIGHT] > min_height]
    
    if v:
        print(f'Selected blobs proportion {len(sel_blobs)/n_labels*100:.2f}%')

    y_shift = 0
    rows_pos = np.empty((0,2), float)
    init_rows_pos = np.empty((0,2), float)
    row_blobs = []

    x_records = []
    y_shift_records = []

    for blob_id in tqdm(left_order):
        if blob_id not in sel_blobs:
            continue

        top_blob = stats[blob_id,cv2.CC_STAT_TOP]
        bot_blob = top_blob + stats[blob_id,cv2.CC_STAT_HEIGHT]
        shifted_top_blob = top_blob - y_shift
        shifted_bot_blob = bot_blob - y_shift

        intersections = []
        for top_line, bot_line in rows_pos:
            top_inter = max(top_line, shifted_top_blob)
            bot_inter = min(bot_line, shifted_bot_blob)
            intersections.append(max(0, bot_inter-top_inter))

        if any([inter > 0 for inter in intersections]):
            row_inter = np.argmax(intersections)
            top_line, bot_line = rows_pos[row_inter]

            rows_pos[row_inter] = shifted_top_blob, shifted_bot_blob 
            row_blobs[row_inter].append(blob_id)

            bot_diff = bot_blob - bot_line
            y_shift = alpha * y_shift + (1-alpha)*bot_diff

            y_shift_records.append(y_shift)
            x_records.append(stats[blob_id, cv2.CC_STAT_LEFT])

        else:
            rows_pos = np.vstack([rows_pos, [shifted_top_blob, shifted_bot_blob]])
            init_rows_pos = np.vstack([init_rows_pos, [shifted_top_blob, shifted_bot_blob]])
            row_blobs.append([blob_id])

    row_order = np.argsort(rows_pos[:,0])
    rows_pos = rows_pos[row_order]
    init_rows_pos = init_rows_pos[row_order]
    row_blobs = [row_blobs[i] for i in row_order]

    X = np.array(x_records).reshape((-1,1))
    ransac = RANSACRegressor().fit(X, y_shift_records)
    skew_per_px = ransac.estimator_.coef_
    angle = np.arctan(skew_per_px)[0] * 180/np.pi

    # TODO: the line positions are rotated as well ...
    M = cv2.getRotationMatrix2D((0,0), angle, 1.0)
    h, w = img.shape
    rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated_row_pos = []  # could be optimized
    for top_row, bot_row in init_rows_pos:
        rot_top_row = int(top_row*M[1,1] + M[1,2])
        rot_bot_row = int(bot_row*M[1,1] + M[1,2])
        rotated_row_pos.append([rot_top_row, rot_bot_row])
    rotated_row_pos = np.array(rotated_row_pos)

    if v:
        print(f'Number of detected rows {len(rows_pos)}')
        plt.plot(x_records, y_shift_records, '--x')

        plt.plot(x_records, ransac.predict(X))
        plt.show()

    if v >1:
        col_img = get_color(img)
        colors = [(255,0,0), (0,255,0), (0,0,255)]
        for i, row_id in tqdm(enumerate(row_order)):
            blob_ids = row_blobs[row_id] 
            color = colors[i%len(colors)]
            for blob_id in blob_ids:
                col_img[labels == blob_id] = color
        show([col_img], 0.2)

    # row_boxes = []
    # for blob_ids in row_blobs:
    #     top = stats[blob_ids, cv2.CC_STAT_TOP].min()
    #     bot = max([stats[blob_id, cv2.CC_STAT_HEIGHT] + stats[blob_id, cv2.CC_STAT_TOP] for blob_id in blob_ids])
    #     row_boxes.append([top, bot])

    return 255-rotated_img, rotated_row_pos

# https://stackoverflow.com/questions/44047819/increase-image-brightness-without-overflow/44054699#44054699
def rm_shadow(img):
    dilated_img = cv2.dilate(img, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(img, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return diff_img, norm_img


# OLD
# Crops up and down if there is "a lot of black"
def crop_height(gray_img, dpi, th=120, min_black=20):
    ymin, ymax = 0, gray_img.shape[0]
    hsum = gray_img.mean(axis=1)
    is_black_line = hsum < th
    is_white_line = hsum > th
    if is_black_line[:min_black].all():
        ymin = is_white_line.argmax() + int(TOP_CROP_ADD_IN * dpi)
    if is_black_line[ymax-min_black].all():
        ymax -= np.flip(is_white_line).argmax() + int(BOTTOM_CROP_ADD_IN * dpi)
    return gray_img[ymin:ymax]