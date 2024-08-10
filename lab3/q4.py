import cv2
import numpy as np
def sobel_edge_detection(image):
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
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def laplacian_edge_detection(image):
    laplacian_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)

    laplacian = cv2.filter2D(image, cv2.CV_64F, laplacian_kernel)
    return np.clip(np.abs(laplacian), 0, 255).astype(np.uint8)

if __name__ == "__main__":
    img = cv2.imread("input.png")
    sobel_edges = sobel_edge_detection(img)
    laplacian_edges = laplacian_edge_detection(img)
    cv2.imshow("Sobel Edges", sobel_edges)
    cv2.imshow("Laplacian Edges", laplacian_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
