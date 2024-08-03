import cv2
image = cv2.imread("computer_vision.png")
print(image.dtype)
x,y,z = image.shape
for i in range(10):
    for j in range(10):
        print("pixel number",i,",",j,"is",image[i][j])