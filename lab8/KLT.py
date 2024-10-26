import cv2
import numpy as np

def detect_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    return corners

def track_features(prev_img, curr_img, corners):
    gray1 = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    next_corners, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners, None)
    return next_corners, status

def visualize_tracking(img, corners, next_corners, status):
    for i, (new, old) in enumerate(zip(next_corners, corners)):
        if status[i] == 1:
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            cv2.line(img, (a, b), (c, d), (0, 255, 0), 2)
            cv2.circle(img, (a, b), 5, (0, 0, 255), -1)
    return img

def main():
    cap = cv2.VideoCapture('/home/student/Documents/220962101/Week8/sunset.mp4')
    ret, old_frame = cap.read()
    if not ret:
        return

    corners = detect_corners(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        next_corners, status = track_features(old_frame, frame, corners)

        if next_corners is not None and status is not None:
            if np.sum(status) < len(status) / 2:
                corners = detect_corners(frame)

            output_frame = visualize_tracking(frame.copy(), corners, next_corners, status)
            cv2.imshow('KLT Tracker', output_frame)

            old_frame = frame.copy()
            corners = next_corners[status.flatten() == 1]

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
