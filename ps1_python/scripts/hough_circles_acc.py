import numpy as np

def hough_circles_acc(edges, rad):
    h, w = edges.shape
    accumulator = np.zeros(edges.shape, dtype=np.uint8)
    thetas = np.deg2rad(np.arange(0, 361, 1))
    y_idxs, x_idxs = np.nonzero(edges)
    
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for theta in thetas:
            a = int(x - rad * np.cos(theta))
            b = int(y + rad * np.sin(theta))
            if a < h and b < w: 
                accumulator[a][b] += 1
    return accumulator
