import cv2
import numpy as np
def generate_random_gaussian_kernel(size, sigma):
    random_matrix = np.random.rand(size, size)
    center = size // 2
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) *
                     np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    kernel = random_matrix * kernel
    return kernel / np.sum(kernel)

if __name__ == "__main__":
    img = cv2.imread("input.png",0)
    kernel_size = 5
    sigma = 1.0
    gaussian_kernel = generate_random_gaussian_kernel(kernel_size, sigma)
    blur = cv2.filter2D(img, -1, gaussian_kernel)
    mask = cv2.subtract(img, blur)
    sharpened = cv2.addWeighted(img, 1.0 + 1.2, mask, -1.2, 0)
    cv2.imshow("unsharp",sharpened)
    cv2.imshow("original",img)
    cv2.imshow("gauss_blur",blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
