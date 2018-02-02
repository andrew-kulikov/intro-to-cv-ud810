import numpy as np
import cv2
from matplotlib import pyplot as plt
from disparity_ssd import *

def main():
    L = cv2.imread('input/pair1-L.png', cv2.IMREAD_GRAYSCALE)
    R = cv2.imread('input/pair1-R.png', cv2.IMREAD_GRAYSCALE)
    
    #some gaussian noise
    cols, rows = L.shape
    noise = np.random.randn(cols, rows)
    
    L_noise = L + np.uint8(noise * 10)
    R_noise = R + np.uint8(noise * 10)
    
    #disparity maps for noised images
    disparity_left = np.abs(disparity_ssd(L_noise, R_noise, window_size=15, disparity_range=200))
    disparity_right = np.abs(disparity_ssd(R_noise, L_noise, window_size=15, disparity_range=200))
    
    disparity_left = cv2.normalize(disparity_left, disparity_left, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disparity_right = cv2.normalize(disparity_right, disparity_right, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    cv2.imwrite('output/ps2-3-a-1.png', disparity_left)
    cv2.imwrite('output/ps2-3-a-2.png', disparity_right)
    
    #increase contrast
    L_contrast = np.uint8(L * 1.1)
    R_contrast = np.uint8(R * 1.1)
    
    #disparity maps for contrast images
    disparity_left = np.abs(disparity_ssd(L_contrast, R_contrast, window_size=15, disparity_range=200))
    disparity_right = np.abs(disparity_ssd(R_contrast, L_contrast, window_size=15, disparity_range=200))
    
    disparity_left = cv2.normalize(disparity_left, disparity_left, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disparity_right = cv2.normalize(disparity_right, disparity_right, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imwrite('output/ps2-3-b-1.png', disparity_left)
    cv2.imwrite('output/ps2-3-b-2.png', disparity_right)
    
    
    

if __name__ == '__main__':
    main()
