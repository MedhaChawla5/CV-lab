import cv2
image = cv2.imread("computer_vision.png")
window_name = 'question_4'
start_point = (72, 700)
end_point = (270, 300)
color = (0, 255, 0)
thickness = 2
image = cv2.rectangle(image, start_point, end_point, color, thickness)
cv2.imshow(window_name, image)
cv2.waitKey(0)