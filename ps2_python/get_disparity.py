import cv2
import numpy as np
from disparity_ssd import *
from disparity_ncorr import *


def get_disparity(img_left, img_right, mode='ssd', window_size=5, disparity_range=50):
    if mode == 'ssd':
        disparity_left = np.abs(disparity_ssd(img_left, img_right, window_size=window_size, disparity_range=disparity_range))
        disparity_right = np.abs(disparity_ssd(img_right, img_left, window_size=window_size, disparity_range=disparity_range))
    elif mode == 'ncorr':
        disparity_left = np.abs(disparity_ncorr(img_left, img_right, window_size=window_size, disparity_range=disparity_range))
        disparity_right = np.abs(disparity_ncorr(img_right, img_left, window_size=window_size, disparity_range=disparity_range))
    
    disparity_left = cv2.normalize(disparity_left, disparity_left, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disparity_right = cv2.normalize(disparity_right, disparity_right, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return disparity_left, disparity_right
