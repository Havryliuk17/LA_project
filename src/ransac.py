import numpy as np
from random import sample

def compute_homography_matrix(source_points, destination_points):
    """Compute the homography matrix using the direct linear transformation (DLT) method"""
    assert len(source_points) == len(destination_points), "Source and destination points must be of same length"
    num_points = len(source_points)
    matrix_A = np.zeros((2 * num_points, 9))
    for index, (src, dst) in enumerate(zip(source_points, destination_points)):
        x, y = src
        u, v = dst
        matrix_A[2 * index] = [x, y, 1, 0, 0, 0, -u * x, -u * y, -u]
        matrix_A[2 * index + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y, -v]
    _, _, Vt = np.linalg.svd(matrix_A)
    homography = Vt[-1].reshape(3, 3)
    return homography / homography[2, 2]

def transform_point(point, homography):
    """Applies the homography transformation to a point"""
    x, y = point
    denominator = homography[2, 0] * x + homography[2, 1] * y + homography[2, 2]
    transformed_x = (homography[0, 0] * x + homography[0, 1] * y + homography[0, 2]) / denominator
    transformed_y = (homography[1, 0] * x + homography[1, 1] * y + homography[1, 2]) / denominator
    return [transformed_x, transformed_y]

def count_inliers(points_src, points_dst, homography, threshold):
    """Count the number of inliers based on the reprojection error"""
    inliers_count = 0
    for src_point, dst_point in zip(points_src, points_dst):
        predicted_point = transform_point(src_point, homography)
        error = np.linalg.norm(np.array(predicted_point) - np.array(dst_point))
        if error < threshold:
            inliers_count += 1
    return inliers_count

def ransac(src_points, dst_points, samples=4, iterations=5000, tolerance=3):
    """Apply the RANSAC algorithm to estimate a homography matrix from source to destination points"""
    best_homography = None
    maximum_inliers = 0
    best_inliers = []
    for _ in range(iterations):
        indices = sample(range(len(src_points)), samples)
        src_samples = [src_points[i] for i in indices]
        dst_samples = [dst_points[i] for i in indices]
        current_homography = compute_homography_matrix(src_samples, dst_samples)
        current_inliers = count_inliers(src_points, dst_points, current_homography, tolerance)
        if current_inliers > maximum_inliers:
            maximum_inliers = current_inliers
            best_homography = current_homography
            best_inliers = [i for i in range(len(src_points)) if np.linalg.norm(np.array(transform_point(src_points[i], current_homography)) - np.array(dst_points[i])) < tolerance]
    return best_homography, best_inliers
