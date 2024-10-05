#ratio test
import cv2
import numpy as np

def ratio_test(matches, ratio=0.75):
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches

# Example pipeline
def main():
    # Load images
    img1 = cv2.imread('img_1.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('img_3.png', cv2.IMREAD_GRAYSCALE)

    # Step 1: Detect keypoints and descriptors using SIFT
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Step 2: Match descriptors using KNN
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)  # k=2 to get two nearest neighbors

    # Step 3: Apply the ratio test
    good_matches = ratio_test(matches)

    # Draw matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show results
    cv2.imshow('Good Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
