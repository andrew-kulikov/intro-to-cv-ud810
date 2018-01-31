import numpy as np
import cv2
from hough_circles_acc import *
from hough_circles_draw import *
from hough_peaks import *
from find_circles import *

def run():
    img = cv2.imread('../input/ps1-input1.png', cv2.IMREAD_GRAYSCALE)
    img_smoothed = cv2.GaussianBlur(img, (31, 31), 3)
    cv2.imwrite('../output/ps1-5-a-1.png', img_smoothed)
    
    edges_smoothed = cv2.Canny(img_smoothed, 0, 80)
    cv2.imwrite('../output/ps1-5-a-2.png', edges_smoothed)
    
    accumulator = np.uint8(hough_circles_acc(edges_smoothed, 20))
    cv2.imwrite('../output/accum.png', accumulator)
    centers = np.int16(hough_peaks(accumulator, 10, 150))
    rads = np.int16(np.zeros(centers.shape[0]) + 20)
    img1 = np.copy(img)
    hough_circles_draw(img1, '../output/ps1-5-a-3.png', centers, rads)
    
    centers1, rads1 = find_circles(edges_smoothed, (20, 50), 150)
    img2 = np.copy(img)
    hough_circles_draw(img2, '../output/ps1-5-b-1.png', np.int16(centers1), np.int16(rads1))
    

if __name__ == '__main__':
    run()