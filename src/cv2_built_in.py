import cv2
import numpy as np
### We used this code for comparasion 
def stitch_images(img1, img2):
    """Stitches images using built-in functions"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    img1_kp = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_kp = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    img_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    img_good_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
    
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)
    
    for i, match in enumerate(good_matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
    

    H, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    
    inliers = [good_matches[i] for i in range(len(mask)) if mask[i]]
    img_inliers = cv2.drawMatches(img1, kp1, img2, kp2, inliers, None, flags=2)
    
    height, width, channels = img1.shape
    img2_transformed = cv2.warpPerspective(img2, H, (width * 2, height))
    

    panorama = img2_transformed.copy()
    panorama[0:height, 0:width] = img1
    

    cv2.imshow('KeyPoints 1', img1_kp)
    cv2.imshow('KeyPoints 2', img2_kp)
    cv2.imshow('All Matches', img_matches)
    cv2.imshow('Good Matches', img_good_matches)
    cv2.imshow('Inliers after RANSAC', img_inliers)
    cv2.imshow('Panorama', panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return panorama

img1 = cv2.imread('path_to_image1.jpg')
img2 = cv2.imread('path_to_image2.jpg')

result = stitch_images(img1, img2)
