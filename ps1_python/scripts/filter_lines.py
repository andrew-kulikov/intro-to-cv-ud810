import numpy as np


def filter_lines(peaks, rhos, thetas, rho_threshold, theta_threshold):
    good_peaks = []
    for i in range(peaks.shape[0] - 1):
        delta_rho = np.array([abs(rhos[peaks[i][0]] - rhos[peaks[j][0]]) for j in range(peaks.shape[0] - 1)])
        delta_theta = np.array([abs(thetas[peaks[i][1]] - thetas[peaks[j][1]]) for j in range(peaks.shape[0] - 1)])
        if ((delta_rho > 0) & (delta_rho < rho_threshold) & (delta_theta < theta_threshold)).any():
            good_peaks += [i]
    return peaks[good_peaks]
        
