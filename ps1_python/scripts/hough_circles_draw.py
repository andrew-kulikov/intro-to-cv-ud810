import numpy as np
import cv2

def hough_circles_draw(img, path, centers, rads):
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(len(centers)):
        y, x = centers[i]
        rad = rads[i]
        cv2.circle(color_img, (y, x), rad, (0, 255, 255), 2)
    cv2.imwrite(path, color_img)