import numpy as np


def least_squares_proj_matrix(points_3d, points_2d):
    #n - amount of points
    n = points_3d.shape[0]
    A = np.zeros((2 * n, 11), dtype=np.float32)
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]
    zeros = np.zeros(n, dtype=np.float32)
    ones = zeros + 1
    A[::2, :] = np.column_stack((X, Y, Z, ones, zeros, zeros, zeros, zeros, -x * X, -x * Y, -x * Z))
    A[1::2, :] = np.column_stack((zeros, zeros, zeros, zeros, X, Y, Z, ones,  -y * X, -y * Y, -y * Z))
    b = np.zeros(2 * n, dtype=np.float32)
    b[::2] = x
    b[1::2] = y
    M, res, _, _ = np.linalg.lstsq(A, b)
    M = np.append(M, 1)
    M = M.reshape((3, 4))
    return M, res
