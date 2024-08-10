import cv2
import numpy as np

def box_kernel(size):
    return np.ones((size, size), dtype=np.float32) / (size * size)

def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) *
                     np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

if __name__ == "__main__":
    img = cv2.imread("input.png")
    box_size = 5
    gaussian_sigma = 1.5
    box_k = box_kernel(box_size)
    gaussian_k = gaussian_kernel(2 * int(2 * gaussian_sigma) + 1, gaussian_sigma)
    gauss = cv2.filter2D(img, -1, gaussian_k)
    box = cv2.filter2D(img,-1,box_k)
    cv2.imshow("box_f",box)
    cv2.imshow('gauss_f',gauss)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
