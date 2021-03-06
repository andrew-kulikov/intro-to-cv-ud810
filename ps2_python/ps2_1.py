import numpy as np
import cv2
from disparity_ssd import *

def main():
    L = cv2.imread('input/pair0-L.png', cv2.IMREAD_GRAYSCALE)
    R = cv2.imread('input/pair0-R.png', cv2.IMREAD_GRAYSCALE)
    disparity_left = disparity_ssd(L, R, window_size=11)
    disparity_right = disparity_ssd(R, L, window_size=11)
    disparity_left = cv2.normalize(disparity_left, disparity_left, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    disparity_right = cv2.normalize(disparity_right, disparity_right, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite('output/ps2-1-a-1.png', disparity_left)
    cv2.imwrite('output/ps2-1-a-2.png', disparity_right)
    
    
    

if __name__ == '__main__':
    main()