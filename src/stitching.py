import numpy as np
import cv2
from knn import knn_matcher, cross_check
from ransac import ransac
import matplotlib.pyplot as plt

def visualize_keypoints(image, keypoints):
    """Draw keypoints detected in an image"""
    keypoint_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return keypoint_image

def visualize_matches(img1, kp1, img2, kp2, matches):
    """Draw matches between keypoints from two images"""
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return match_img

def visualize_inliers(img1, kp1, img2, kp2, matches, inliers):
    """Visualize only the inliers from the matches between keypoints in two images"""
    inlier_matches = [matches[i] for i in inliers]
    cv_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=inlier_matches[i][1], _imgIdx=0, _distance=inlier_matches[i][2]) for i in range(len(inlier_matches))]
    inlier_img = cv2.drawMatches(img1, kp1, img2, kp2, cv_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return inlier_img

def trim_black_borders(image):
    """Trim the black borders from a stitched image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea) 
        x, y, w, h = cv2.boundingRect(cnt)
        return image[y:y+h, x:x+w]
    return image

def stitch_images(img1, img2, rotate=False, visualize=False):
    """Stitch two images together using feature matching and homography, with trimming of black borders"""
    if rotate:
        img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
        img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if visualize:
        img1_kp = visualize_keypoints(img1, kp1)
        img2_kp = visualize_keypoints(img2, kp2)
        cv2.imshow("Keypoints Image 1", img1_kp)
        cv2.imshow("Keypoints Image 2", img2_kp)
        cv2.waitKey(0)

    matches1 = knn_matcher(des1, des2)
    matches2 = knn_matcher(des2, des1)
    good_matches = cross_check(matches1, matches2)

    if visualize:
        dmatches = [cv2.DMatch(_queryIdx=i1, _trainIdx=i2, _distance=dist) for (i1, i2, dist) in good_matches]
        matches_img = visualize_matches(img1, kp1, img2, kp2, dmatches)
        cv2.imshow("Matches after KNN", matches_img)
        cv2.waitKey(0)

    points1 = np.array([kp1[m[0]].pt for m in good_matches], dtype=np.float32)
    points2 = np.array([kp2[m[1]].pt for m in good_matches], dtype=np.float32)

    H, inliers = ransac(points2, points1)

    if visualize:
        inliers_img = visualize_inliers(img1, kp1, img2, kp2, matches1, inliers)
        cv2.imshow("Inliers after RANSAC", inliers_img)
        cv2.waitKey(0)

    img2_transformed = cv2.warpPerspective(img2, H, (img1.shape[1] * 2, img1.shape[0]))
    panorama = img2_transformed.copy()
    panorama[0:img1.shape[0], 0:img1.shape[1]] = img1

    if rotate:
        panorama = cv2.rotate(panorama, cv2.ROTATE_90_COUNTERCLOCKWISE)

    trimmed_panorama = trim_black_borders(panorama)
    return trimmed_panorama

def read_input_and_stitch():
    """Read input image paths from the user, determine the stitching direction,
    produce a stitched panorama"""
    print("Enter 'horizontal' or 'vertical' for stitching direction:")
    direction = input().strip().lower()
    if direction not in ['horizontal', 'vertical']:
        print("Invalid direction specified. Defaulting to horizontal.")
        direction = 'horizontal'
    
    print('Enter the paths to the images you want to stitch. Type "done" when finished:')
    images = []
    while True:
        path = input('Path to image: ')
        if path.lower() == 'done':
            break
        images.append(path)
    
    if len(images) < 2:
        print("Please provide at least two images for stitching.")
        return

    rotate = False
    if direction=='vertical':
        rotate=True
    current_image = cv2.imread(images[0])
    for next_image_path in images[1:]:
        next_image = cv2.imread(next_image_path)
        current_image = stitch_images(current_image, next_image, rotate=rotate)

    if current_image is not None:
        cv2.imshow('Panorama', current_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == '__main__':
    read_input_and_stitch()
