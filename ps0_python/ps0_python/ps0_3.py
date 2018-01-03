import cv2
import numpy as np
from os.path import isfile

def get_center(img):
    return img.shape[0] // 2, img.shape[1] // 2

def paste_crop(source, dest):
    centerY, centerX = get_center(source)
    crop = source[centerY - 50 : centerY + 50, centerX - 50 : centerX + 50]
    centerY, centerX = get_center(dest)
    dest[centerY - 50 : centerY + 50, centerX - 50 : centerX + 50] = crop
    return dest

def run():
    img1 = cv2.imread('../output/ps0-2-c-1.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('../input/raspberry.png', cv2.IMREAD_GRAYSCALE)
    #if not isfile('../output/ps0-3-a-1.jpg'):
    ans = paste_crop(img1, img2)
    cv2.imwrite('../output/ps0-3-a-1.jpg', ans)

if __name__ == '__main__':
    run()