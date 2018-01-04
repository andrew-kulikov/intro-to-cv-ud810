import cv2
import numpy as np

def hough_lines_draw(img, path, peaks, rhos, thetas):
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for peak in peaks:
        rho, theta = np.abs(rhos[peak[0]]), thetas[peak[1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(color_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(path, color_img)

__all__ = ['hough_lines_draw']