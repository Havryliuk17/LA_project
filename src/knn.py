import numpy as np
import cv2

def knn_matcher(des1, des2, k=2):
    """Computes k-nearest neighbors for each descriptor in des1 against des2"""
    matches = {i: [] for i in range(len(des1))}
    for i in range(len(des1)):
        distances = []
        for j in range(len(des2)):
            distance = np.linalg.norm(des1[i] - des2[j])
            distances.append((distance, j))
        distances.sort()
        matches[i] = distances[:k]
    return matches


def cross_check(matches1, matches2, ratio=0.8):
    """Performs cross-checking and applies the ratio test on matches"""
    good_matches = []
    for i, matches in matches1.items():
        if len(matches) > 1 and matches[0][0] < ratio * matches[1][0]:
            min_dist, j = matches[0]
            if len(matches2[j]) > 1 and matches2[j][0][1] == i:
                good_matches.append((i, j, min_dist))
    return good_matches
