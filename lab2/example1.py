import cv2
import numpy as np
img = cv2.imread("computer_vision.png")

c = 255/(np.log(1+ np.max(img)))
log_trans = c* np.log(1+img)

log_trans = np.array(log_trans,dtype = np.uint8)
cv2.imshow("log_transformed.jpg",log_trans)
cv2.waitKey(0)