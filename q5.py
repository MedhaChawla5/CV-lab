import cv2
image = cv2.imread('computer_vision.png')
(h, w) = image.shape[:2]
new_width = 100
aspect_ratio = h / w
new_height = int(new_width * aspect_ratio)
resized_image = cv2.resize(image, (new_width, new_height))
cv2.imshow("Resized image",resized_image)
cv2.waitKey(0)