from img_utils import get_color, show
import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


# Removes the little T at the left of the ticket
# TOOD: above th for n consec steps, or based on histogram
def get_T_bounds(img, v=False):
    vmass = img.sum(axis=0)
    border_th = (vmass.max() - vmass.min())/2
    vmin = np.argmax(vmass > border_th)
    vmax = len(vmass) - np.argmax(np.flip((vmass > border_th)))
    if v:
        plt.plot(vmass)
        plt.show()
        
    img = img[:,vmin:vmax]
    vmass = vmass[vmin:vmax]
    if v:
        plt.plot(vmass)
        plt.show()
        plt.hist(vmass, bins=40)
        plt.show()
        
    T_th1 = (vmass.max() - vmass.min())*0.99 + vmass.min()
    T_th2 = (vmass.max() - vmass.min())*0.93 + vmass.min()
    
    i_1 = np.argmax(vmass >T_th1)
    vmass = vmass[i_1:]
    if v:
        plt.plot(vmass)
        plt.show()
    
    i_2 = np.argmax(vmass < T_th2)
    vmass = vmass[i_2:]
    if v:
        plt.plot(vmass)
        plt.show()
    
    i_3 = np.argmax(vmass > T_th1)
    vmass = vmass[i_3:]
    if v:
        plt.plot(vmass)
        plt.show()
        
    return vmin+i_1+i_2+i_3, vmax


# Computes the homography to get a rectified ticket
# Use each border point of the ticket
# cv2.findHomography uses ransac to get robust estimation
def get_ticket_homography(img):
    h,w = img.shape
    lsrc_pts, rsrc_pts = [], [] # left, right
    # TODO all in matrix operation
    for i in range(h):
        vmin = np.argmax(img[i])
        vmax = w - np.argmax(np.flip(img[i]))
        lsrc_pts.append([vmin, i])
        rsrc_pts.append([vmax, i])
    
    src_pts = np.vstack([lsrc_pts, rsrc_pts])
    dst_pts = np.vstack([np.zeros(h), range(h)])
    dst_pts = np.hstack([dst_pts, np.vstack([np.ones(h)*w, range(h)])]).T

    return cv2.findHomography(src_pts, dst_pts)


# Might be useful to improve


def pca_rotate(bin_img):
    height, width = bin_img.shape
    center = (height-1)/2,(width-1)/2
    out = []
    for i in tqdm(range(height)):
        for j in range(width):
            if bin_img[i,j]:
                out.append((i-center[0],j-center[1]))
    img_pos = np.array(out, dtype=int)
    print('Fitting PCA')
    pca_fit = PCA(2).fit(img_pos)
    print('PCA fitted')
    x_h, x_w = pca_fit.components_ # Should have unit L2 norm
    if np.abs(x_h[0]) < np.abs(x_w[0]):
        x_h, x_w = x_w, x_h
    if x_h[0] < 0:
        x_h *= -1
    theta = np.arccos(x_h[0])
    print('Theta: ', theta)
    rot = cv2.getRotationMatrix2D(center, theta, 1)
    return cv2.warpAffine(bin_img, rot,(width, height))

# rot_img = pca_rotate(th_img)


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



