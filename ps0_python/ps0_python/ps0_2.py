import cv2
import numpy as np
from os.path import isfile


def swap_red_and_blue(img):
    red = img[:, :, 0]
    blue = img[:, :, 2]
    img[:, :, 0] = blue
    img[:, :, 2] = red
    return img

def select_green_channel(img):
    return img[:, :, 1]

def select_red_channel(img):
    return img[:, :, 0]

def run():
    img = cv2.imread('../input/boy.jpg', cv2.IMREAD_COLOR)
    
    #if not isfile('../output/ps0-2-a-1.jpg'):
    img1 = swap_red_and_blue(img)
    cv2.imwrite('../output/ps0-2-a-1.jpg', img1)
    
    #if not isfile('../output/ps0-2-b-1.jpg'):
    img1 = select_green_channel(img)
    cv2.imwrite('../output/ps0-2-b-1.jpg', img1)
        
    #if not isfile('../output/ps0-2-c-1.jpg'):
    img1 = select_red_channel(img)
    cv2.imwrite('../output/ps0-2-c-1.jpg', img1)
    
    #green channel is better
    
if __name__ == '__main__':
    run()