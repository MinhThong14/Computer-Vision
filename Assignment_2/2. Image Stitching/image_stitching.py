import cv2
import numpy as np

def compute_point_img(left_img, right_img):
    """
    Function compute and return matrix conatinning matching points between left image and right image

    :param left_img: left image
    :param right_img: right image
    :return points_left, points_right: matched points of left and right images

    """
    # Create akaze feature detector
    akaze = cv2.AKAZE_create()
    # Detect and compute keypoints and descriptors in left and right image   
    left_key_point, left_descriptor = akaze.detectAndCompute(left_img, None)
    right_key_point, right_descriptor = akaze.detectAndCompute(right_img, None)
    # Create descriptor matcher
    dsc_matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    # Compute matched descriptors
    matches = dsc_matcher.match(left_descriptor, right_descriptor)
    # Sorted matched descriptor in ascending order
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:int(len(matches) * 0.2)]
    # Shaping left and right matched points matrix
    points_left = np.float32([left_key_point[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points_right = np.float32([right_key_point[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return points_left, points_right

def compute_homography(points_right, points_left):
    """
    Function that compute homography

    :param points_right: matched points from right image
    :param points_left: matched points from left image
    :return homography
    """
    homography, _ = cv2.findHomography(points_right, points_left, cv2.RANSAC, 5.0)
    return homography

def warp_image(right_img, left_img, homography):
    """
    Function that compute warped right image

    :param right_img: right image
    :param left_img: left image
    :param homography: homography
    :return right_imag_warped: warped right image corresponded to left image
    """
    right_img_warped = cv2.warpPerspective(right_img, homography, (left_img.shape[1], left_img.shape[0]))
    return right_img_warped

def merge_images(left_img, warped_right_img):
    """
    Function that merge left and warped right image using bitwise_or
    
    :param left_img: left image
    :param warped_right_img: warped right image
    :return merged: merged image
    """
    merged = cv2.bitwise_or(left_img, warped_right_img)
    return merged

if __name__ == "__main__":
    # Load left and right images
    left_img = cv2.imread('./Input_Image/large2_uttower_left.jpg', 0)
    right_img = cv2.imread('./Input_Image/uttower_right.jpg', 0)
    # Compute features and matched arrays
    points_left, points_right = compute_point_img(left_img, right_img)
    # Compute homography
    homography = compute_homography(points_right, points_left)
    # Find the the warped right image
    warped_right_img = warp_image(right_img, left_img, homography)
    # Merge warped right image to the left image
    merged_image = merge_images(left_img, warped_right_img)
    # Diplay the merged image
    cv2.imshow('merged', merged_image)
    cv2.waitKey()
    cv2.destroyAllWindows()