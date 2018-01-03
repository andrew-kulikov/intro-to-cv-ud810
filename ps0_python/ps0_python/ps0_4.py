import cv2
import numpy as np

def get_min_max_pixel_val(img):
    return (np.min(img), np.max(img))

def get_median_pixel(img):
    return np.median(img)

def get_standard_deviation(img):
    return np.std(img)

def magic(img):
    return (img - np.mean(img)) // np.std(img) * 10 + np.mean(img) 

def shift_by_n_pixels_left(img, n):
    rows, cols = img.shape
    M = np.float32([[1, 0, -n], [0, 1, 0]])
    return cv2.warpAffine(img, M, (cols, rows))

def run():
    img = cv2.imread('../output/ps0-2-b-1.jpg', cv2.IMREAD_GRAYSCALE)
    min_val, max_val = get_min_max_pixel_val(img)
    med = get_median_pixel(img)
    std = get_standard_deviation(img)
    print('Min =', min_val, 'Max =', max_val)
    print('Median = ', med)
    print('Std deviation = ', std)
    
    img1 = magic(img)
    cv2.imwrite('../output/ps0-4-b-1.jpg', img1)
    shifted = shift_by_n_pixels_left(img, 2)
    cv2.imwrite('../output/ps0-4-c-1.jpg', shifted)
    diff = cv2.subtract(img, shifted)
    cv2.imwrite('../output/ps0-4-d-1.jpg', diff)

if __name__ == '__main__':
    run()