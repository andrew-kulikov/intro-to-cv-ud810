import cv2
import numpy as np

def run():
    img1 = cv2.imread('../input/boy.jpg', cv2.IMREAD_COLOR)
    img2 = cv2.imread('../input/raspberry.png', cv2.IMREAD_COLOR)
    cv2.imwrite('../output/ps0-1-a-1.jpg', img1)
    cv2.imwrite('../output/ps0-1-a-2.png', img2)
    
if __name__ == '__main__':
    run()