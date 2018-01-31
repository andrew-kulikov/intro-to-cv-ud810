import numpy as np
from hough_circles_acc import *
from hough_peaks import *

def find_circles(edges, radius_range, threshold=100, nhood_size=5):
    all_rads = np.arange(radius_range[0], radius_range[1])
    good_rads = []
    centers = []
    npeaks = 0
    for i in range(len(all_rads)):
        H = hough_circles_acc(edges, all_rads[i])
        peaks = hough_peaks(H, npeaks=10, threshold=threshold, nhood_size=nhood_size)
        for peak in peaks:
            centers.append(peak)
            good_rads.append(all_rads[i])
            npeaks += 1
            print('Progress:', i / len(all_rads) * 100, '%')
    print('peaks detected:', npeaks)
    return np.array(centers), np.array(good_rads)
            
        
    
