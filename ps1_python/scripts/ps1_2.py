import cv2
import numpy as np
from hough_lines_acc import *
from hough_peaks import *
from hough_lines_draw import *
from matplotlib import pyplot as plt

def plot_hough_acc(H, plot_title='Hough Accumulator Plot'):
    ''' A function that plot a Hough Space using Matplotlib. '''
    fig = plt.figure(figsize=(10, 100))
    fig.canvas.set_window_title(plot_title)
    plt.xticks(np.arange(0, H.shape[1], 10))
    plt.imshow(H, cmap='jet')

    plt.xlabel('Theta Direction'), plt.ylabel('Rho Direction')
    plt.tight_layout()
    plt.savefig('testplot.png')
    plt.show()

def run():
    img = cv2.imread('../input/ps1-input0.png', cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    #task a - acuumulator array for hough transform
    accumulator, rhos, thetas = hough_lines_acc(edges)
    
    cv2.imwrite('../output/ps1-2-a-1.png', accumulator)
    #task b - peaks detection
    peaks = np.uint16(hough_peaks(accumulator, 10))
    print(*peaks, sep='\n')
    H_peaks = cv2.cvtColor(np.uint8(accumulator), cv2.COLOR_GRAY2BGR)
    for peak in peaks:
        cv2.circle(H_peaks, (peak[1], peak[0]), 2, (0, 0, 255))
    cv2.imwrite('../output/ps1-2-b-1.png', H_peaks)
    #task 3 - draw lines
    hough_lines_draw(img, '../output/ps1-2-c-1.png', peaks, rhos, thetas)
    
    

if __name__ == '__main__':
    run()