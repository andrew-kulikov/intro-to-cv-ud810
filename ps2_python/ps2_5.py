import numpy as np
import cv2
from matplotlib import pyplot as plt
from get_disparity import *

def main():
    L = cv2.imread('input/pair2-L.png', cv2.IMREAD_GRAYSCALE)
    R = cv2.imread('input/pair2-R.png', cv2.IMREAD_GRAYSCALE)
    
    #blured images
    L_blured = cv2.GaussianBlur(L, (31, 31), 0.7)
    R_blured = cv2.GaussianBlur(R, (31, 31), 0.7)
    """
    #disparity maps for blured images
    disparity_left, disparity_right = get_disparity(L_blured, R_blured, 'ncorr', 15, 200)
    
    cv2.imwrite('output/ps2-5-a-1.png', disparity_left)
    cv2.imwrite('output/ps2-5-a-2.png', disparity_right)
    """
    #sharped images
    L_sharped = 2 * L - L_blured
    R_sharped = 2 * R - R_blured
    
    cv2.imwrite('output/sharp.png', L_sharped)
    
    

if __name__ == '__main__':
    main()
