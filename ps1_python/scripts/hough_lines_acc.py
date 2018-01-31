import numpy as np

def hough_lines_acc(edges, theta = (-90, 90), theta_res = 1, rho_res = 1):
    height, width = edges.shape
    thetas = np.deg2rad(np.arange(theta[0], theta[1], theta_res, dtype=np.int16))
    max_rho = np.ceil(np.sqrt(height**2 + width**2))
    rhos = np.arange(-max_rho, max_rho + 1, rho_res)
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(edges)
    
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for j in range(len(thetas)):
            rho = int(x * np.cos(thetas[j]) + y * np.sin(thetas[j]) + max_rho)
            H[rho][j] += 1
    
    return H, rhos, thetas
            
    


