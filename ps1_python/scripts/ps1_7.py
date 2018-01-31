import numpy as np
import cv2
from hough_circles_acc import *
from hough_circles_draw import *
from hough_peaks import *
from find_circles import *

def run():
    img = cv2.imread('../input/ps1-input2.png', cv2.IMREAD_GRAYSCALE)
    img_smoothed = cv2.GaussianBlur(img, (31, 31), 3)

    
    edges_smoothed = cv2.Canny(img_smoothed, 0, 80)
    
    
    
    centers, rads = find_circles(edges_smoothed, (25, 50), 100)
    img1 = np.copy(img)
    hough_circles_draw(img1, '../output/ps1-7-a-1.png', np.int16(centers), np.int16(rads))
   

if __name__ == '__main__':
    run()