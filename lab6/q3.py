import cv2
import numpy as np

# Function to estimate homography using OpenCV's built-in methods
def estimate_homography(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors using SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Match descriptors using FLANN
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Filter good matches using Lowe's ratio test
    good_matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.7 * m[1].distance]

    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Estimate homography using RANSAC
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    return H

# Load images
img1 = cv2.imread('img_1.png')
img2 = cv2.imread('img_3.png')

# Estimate homography
H = estimate_homography(img1, img2)

# Print the estimated homography matrix
print("Estimated Homography Matrix:")
print(H)
