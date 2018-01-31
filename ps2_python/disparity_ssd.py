import cv2
import numpy as np
from matplotlib import pyplot as plt

def disparity_ssd(L, R, window_size=5, disparity_range=50):
    D = np.zeros(L.shape)
    for i in range(L.shape[0] - window_size):
        for j in range(L.shape[1] - window_size):
            patch = L[i : i + window_size, j : j + window_size]
            left_col = max(j - disparity_range // 2, 0)
            right_col = min(j + disparity_range // 2 - 1, R.shape[1])
            R_strip = R[i : i + window_size, left_col : right_col]
            mse = cv2.matchTemplate(R_strip, patch, cv2.TM_SQDIFF_NORMED)
            _, _, min_loc, _ = cv2.minMaxLoc(mse[0])
            dist = j - left_col - min_loc[1]
            D[i][j] = dist
    return D

