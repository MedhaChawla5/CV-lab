import cv2
import numpy as np
def sobel_gradient(image):
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    sobel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ], dtype=np.float32)
    grad_x = cv2.filter2D(image, cv2.CV_64F, sobel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, sobel_y)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude_display = np.uint8(magnitude)
    cv2.imshow("Gradient Magnitude", magnitude_display)
    cv2.imshow("gradient_x",grad_x)
    cv2.imshow("gradient_y",grad_y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img = cv2.imread("input.png")
    sobel_gradient(img)