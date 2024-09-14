import numpy as np
import cv2
import matplotlib.pyplot as plt

def lbp_img(image):

    lbp_image = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            window = image[i-1:i+2, j-1:j+2]
            center = window[1, 1]
            bin_window = (window >= center).astype(np.uint8)
            vector = bin_window.flatten()
            vector = np.delete(vector, 4)

            decimal = np.where(vector)[0] 
            if len(decimal) >= 1:
                num = np.sum(2**decimal)
            else:
                num = 0

            lbp_image[i, j] = num

    return lbp_image


input_image = cv2.imread('/home/Student/Downloads/image.jpg', cv2.IMREAD_GRAYSCALE)
processed_img = lbp_img(input_image)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(input_image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('LBP Image')
plt.imshow(processed_img, cmap='gray')

plt.show()
