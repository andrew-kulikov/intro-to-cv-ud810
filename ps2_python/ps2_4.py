import numpy as np
import cv2
from matplotlib import pyplot as plt
from get_disparity import *

def main():
    L = cv2.imread('input/pair1-L.png', cv2.IMREAD_GRAYSCALE)
    R = cv2.imread('input/pair1-R.png', cv2.IMREAD_GRAYSCALE)
    
    disparity_left, disparity_right = get_disparity(L, R, 'ncorr', 15, 200)
    
    cv2.imwrite('output/ps2-4-a-1.png', disparity_left)
    cv2.imwrite('output/ps2-4-a-2.png', disparity_right)
    
    #some gaussian noise
    cols, rows = L.shape
    noise = np.random.randn(cols, rows)
    
    L_noise = L + np.uint8(noise * 10)
    R_noise = R + np.uint8(noise * 10)
    
    #disparity maps for noised images
    disparity_left, disparity_right = get_disparity(L_noise, R_noise, 'ncorr', 15, 200)
    
    cv2.imwrite('output/ps2-4-b-1.png', disparity_left)
    cv2.imwrite('output/ps2-4-b-2.png', disparity_right)
    
    #increase contrast
    L_contrast = np.uint8(L * 1.1)
    R_contrast = np.uint8(R * 1.1)
    
    #disparity maps for contrast images
    disparity_left, disparity_right = get_disparity(L_contrast, R_contrast, 'ncorr', 15, 200)
    
    cv2.imwrite('output/ps2-4-b-3.png', disparity_left)
    cv2.imwrite('output/ps2-4-b-4.png', disparity_right)
    
    
    

if __name__ == '__main__':
    main()
