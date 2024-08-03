import cv2
import matplotlib.pyplot as plt
img = cv2.imread('computer_vision.png')
img_cw_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
img_ccw_90 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
img_cw_180 = cv2.rotate(img, cv2.ROTATE_180)

plt.subplot(221), plt.imshow(img, 'gray'), plt.title('Original Image'), plt.axis('off')
plt.subplot(222), plt.imshow(img_cw_90,'gray'), plt.title('(90degree clockwise)'), plt.axis('off')
plt.subplot(223), plt.imshow(img_cw_180, 'gray'),
plt.title('180 degree'), plt.axis('off')
plt.subplot(224), plt.imshow(img_ccw_90, 'gray'),
plt.title('90 degree counterclockwise'), plt.axis('off')
plt.show()