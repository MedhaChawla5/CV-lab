import cv2
img = cv2.imread("computer_vision.png",0)
cv2.imshow("My first cv program",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
