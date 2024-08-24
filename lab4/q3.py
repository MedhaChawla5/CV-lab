import cv2
img = cv2.imread('input.png')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_range = (0, 50, 50)

upper_range = (150, 255, 255)
mask = cv2.inRange(hsv_img, lower_range, upper_range)
color_image = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow('Original Image', img)
cv2.imshow('Coloured Image', color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
