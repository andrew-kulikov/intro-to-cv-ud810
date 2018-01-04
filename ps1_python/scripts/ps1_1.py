import numpy as np
import cv2

def run():
    img = cv2.imread('../input/ps1-input0.png', cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    cv2.imwrite('../output/ps1-1-a-1.png', edges)
    
if __name__ == '__main__':
    run()