import cv2
import numpy as np


def run():
    img = cv2.imread('../input/boy.jpg', cv2.IMREAD_COLOR)
    #img = good_stuff.im2double(img)
    green = img[:, :, 1]
    noise = np.random.randn(green.shape[0], green.shape[1])
    green = green + noise * 20
    green_noised = np.copy(img)
    green_noised[:, :, 1] = green
    cv2.imwrite('../output/ps0-5-a-1.png', green_noised) 
    
    blue_noised = np.copy(img)
    blue_noised[:, :, 2] = blue_noised[:, :, 2] + noise * 20
    cv2.imwrite('../output/ps0-5-b-1.png', blue_noised) 
    
    

if __name__ == '__main__':
    run()