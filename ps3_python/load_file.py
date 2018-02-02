import numpy as np


def load_file(path):
    f = open(path)
    lines = f.readlines()
    M = []
    for line in lines:
        nums = list(map(float, line.split()))
        M .append(nums)
    f.close()
    return np.array(M)
    