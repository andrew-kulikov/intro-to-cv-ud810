import numpy as np
from cv2 import minMaxLoc

def clip(t):
    return max(int(t), 0)

def hough_peaks(H, npeaks=1, threshold=100, nhood_size=5):
    peaks = np.zeros((npeaks, 2))
    tmp_H = np.copy(H)
    for i in range(npeaks):
        _, max_val, _, max_loc = minMaxLoc(tmp_H)
        if max_val > threshold:
            (x, y) = max_loc
            peaks[i] = (y, x)
            
            dist = nhood_size // 2
            tmp_H[clip(y - dist) : y + dist + 1, clip(x - dist) : x + dist + 1] = 0
        else:
            peaks = peaks[:i]
            break
    return peaks

__all__ = ['hough_peaks']
