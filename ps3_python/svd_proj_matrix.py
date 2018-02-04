import numpy as np


def svd_proj_matrix(points_3d, points_2d):
    #n - amount of points
    n = points_3d.shape[0]
    A = np.zeros((2 * n, 12), dtype=np.float32)
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]
    zeros = np.zeros(n, dtype=np.float32)
    ones = zeros + 1
    A[::2, :] = np.column_stack((X, Y, Z, ones, zeros, zeros, zeros, zeros, -x * X, -x * Y, -x * Z, -x))
    A[1::2, :] = np.column_stack((zeros, zeros, zeros, zeros, X, Y, Z, ones,  -y * X, -y * Y, -y * Z, -y))
    _, _, V = np.linalg.svd(A)
    M = V.T[:, -1]
    M = M.reshape((3, 4))
    return M
