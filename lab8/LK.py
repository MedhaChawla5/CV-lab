import cv2
import numpy as np

sift = cv2.SIFT_create()
cap = cv2.VideoCapture('/home/student/Documents/220962101/Week8/sunset.mp4')

ret, old_f = cap.read()
if not ret:
    print("Failed to read the video.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

old_g = cv2.cvtColor(old_f, cv2.COLOR_BGR2GRAY)
kp, des = sift.detectAndCompute(old_g, None)
p0 = cv2.KeyPoint_convert(kp)
mask = np.zeros_like(old_f)

while True:
    ret, f = cap.read()
    if not ret:
        break

    f_g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_g, f_g, p0, None)

    if p1 is not None and st is not None:
        g_new = p1[st.flatten() == 1]
        g_old = p0[st.flatten() == 1]

        if len(g_new) >= 3:
            m, _ = cv2.estimateAffine2D(g_old, g_new)

            for i, (n, o) in enumerate(zip(g_new, g_old)):
                a, b = n.ravel().astype(int)
                c, d = o.ravel().astype(int)
                mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
                f = cv2.circle(f, (a, b), 5, (0, 0, 255), -1)

    old_g = f_g.copy()
    p0 = g_new.reshape(-1, 1, 2)
    img = cv2.add(f, mask)
    cv2.imshow('Optical Flow Tracking', img)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
