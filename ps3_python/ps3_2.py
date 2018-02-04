import numpy as np
import random
from load_file import *
from least_squares_proj_matrix import *
from svd_proj_matrix import *
from check_point import *


def main():
    points_2d = load_file('input/pts2d-norm-pic_a.txt')
    points_3d = load_file('input/pts3d-norm.txt')
    
    repeats = 10
    k_amounts = [8, 12, 16]
    num_residuals = 4
    results = np.zeros((repeats, len(k_amounts)))
    best_M = []
    lowest_residual = np.inf
    
    for i in range(repeats):
        for k in range(len(k_amounts)):
            #for each of k amounts we choose points and train on them ls
            choosen_numbers = random.sample(range(points_2d.shape[0]), k_amounts[k])
            not_choosen_numbers = [x for x in range(points_2d.shape[0]) if x not in choosen_numbers]
            
            choosen_points_2d = points_2d[choosen_numbers]
            choosen_points_3d = points_3d[choosen_numbers]
            not_choosen_points_2d = points_2d[not_choosen_numbers]
            not_choosen_points_3d = points_3d[not_choosen_numbers]
            
            
            #M, c = least_squares_proj_matrix(choosen_points_3d, choosen_points_2d)
            M = svd_proj_matrix(choosen_points_3d, choosen_points_2d)
            residuals = []
            
            #then test given projection matrix on test points
            test_points = random.sample(range(len(not_choosen_numbers)), num_residuals)
            for j in range(num_residuals):
                test_point = test_points[j]
                residuals.append(check_point(not_choosen_points_3d[test_point], not_choosen_points_2d[test_point], M))
            
            average_residual = np.average(residuals)
            if average_residual < lowest_residual:
                lowest_residual = average_residual
                best_M = M
            results[i][k] = average_residual
    
    Q = best_M[:3, :3]
    m = best_M[:, 3]
    camera_center = - np.dot(np.linalg.inv(Q), m)
    print(camera_center)
    
    f = open('output/ps3-1-b-1.txt', 'w+')
    f.write('Best Projection Matrix:\n' + str(best_M) + '\nResult table:\n' + str(results))
    f.close()
    f = open('output/ps3-1-best_m.txt', 'w+')
    f.write(str(best_M).replace('[', '').replace(']', ''))
    f.close()
    
    


if __name__ == '__main__':
    main()