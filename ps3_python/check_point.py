import numpy as np

def check_point(point_3d, point_2d, M):
    point_3d = np.append(point_3d, 1)
    predicted_point = np.dot(M, point_3d)
    predicted_point = predicted_point[:2] / predicted_point[2]
    return np.linalg.norm(predicted_point - point_2d)
