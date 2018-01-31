import cv2
import numpy as np
from hough_lines_acc import *
from hough_peaks import *
from hough_lines_draw import *


def run():
    img = cv2.imread('../input/ps1-input1.png', cv2.IMREAD_GRAYSCALE)
    img_smoothed = cv2.GaussianBlur(img, (31, 31), 3)
    cv2.imwrite('../output/ps1-4-a-1.png', img_smoothed)
    
    edges_smoothed = cv2.Canny(img_smoothed, 0, 80)
    cv2.imwrite('../output/ps1-4-b-1.png', edges_smoothed)
    
    #task с - acuumulator array for hough transform
    accumulator, rhos, thetas = hough_lines_acc(edges_smoothed)
    peaks = np.int16(hough_peaks(accumulator, 4, 70))
    H_peaks = cv2.cvtColor(np.uint8(accumulator), cv2.COLOR_GRAY2BGR)
    for peak in peaks:
        cv2.circle(H_peaks, (peak[1], peak[0]), 2, (0, 0, 255))
    cv2.imwrite('../output/ps1-4-c-1.png', H_peaks)
    
    #draw lines
    hough_lines_draw(img, '../output/ps1-4-c-2.png', peaks, rhos, thetas)
   
    

if __name__ == '__main__':
    run()