import numpy as np
from load_file import *
from least_squares_proj_matrix import *
from check_point import *


def main():
    points_2d = load_file('input/pts2d-norm-pic_a.txt')
    points_3d = load_file('input/pts3d-norm.txt')
    M, c = least_squares_proj_matrix(points_3d, points_2d)
    f = open('output/ps3-1-a-1.txt', 'w+')
    residual = check_point(points_3d[-1], points_2d[-1], M)
    f.write('Projection Matrix:\n' + str(M) + '\nresidual:\n' + str(residual))
    f.close()

    

if __name__ == '__main__':
    main()