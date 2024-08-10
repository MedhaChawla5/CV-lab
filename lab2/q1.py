import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'computer_vision.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Could not read the image.")
    exit()

histogram = np.zeros(256, dtype=int)
height, width = image.shape

for i in range(height):
    for j in range(width):
        pixel_value = image[i, j]
        histogram[pixel_value] += 1
cdf = np.cumsum(histogram)

cdf_min = cdf[cdf > 0].min()
cdf_range = cdf.max() - cdf_min
cdf_normalized = (cdf - cdf_min) * 255 / cdf_range
cdf_normalized = np.round(cdf_normalized).astype(int)

equalized_image = np.zeros_like(image)
for i in range(height):
    for j in range(width):
        pixel_value = image[i, j]
        equalized_image[i, j] = cdf_normalized[pixel_value]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Equalized Image')
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

plt.show()
