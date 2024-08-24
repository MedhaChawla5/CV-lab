import cv2 as cv
import numpy as np

def kmeans_color_segmentation(image_path, k=3):
    src = cv.imread(cv.samples.findFile(image_path))

    if src is None:
        print("Error opening image!")
        return
    img_rgb = cv.cvtColor(src, cv.COLOR_BGR2RGB)
    pixel_values = img_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv.kmeans(pixel_values, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img_rgb.shape)
    segmented_image_bgr = cv.cvtColor(segmented_image, cv.COLOR_RGB2BGR)
    cv.imshow('Original Image', src)
    cv.imshow('Segmented Image', segmented_image_bgr)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    image_path = 'img_2.png'
    k = 3
    kmeans_color_segmentation(image_path, k)
