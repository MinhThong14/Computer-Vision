import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import copy


def gaussian_blur(img0, sigma):
    """
    Function that smoothing the image using gaussian blur

    :param img0: input image
    :param sigma: standard deviation
    :return: the smoothing image after applying gaussian kernel

    """
    # hsize
    hsize = 2 * int(math.ceil(3 * sigma)) + 1
    # gaussian blur
    img0 = cv2.GaussianBlur(img0, (hsize, hsize), sigma)
    return img0

def sobel_filters(img0):
    """
    Function that create sobel filters

    :params img0: input image
    :return image gradients in the x and y direction

    """
    # Sobel filter for gradient in x direction
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    imgx = cv2.filter2D(img0, cv2.CV_32F, sobel_x)
    
    # Sobel filter for gradient in y direction
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    imgy = cv2.filter2D(img0, cv2.CV_32F, sobel_y)
    
    return imgx, imgy

def gradient_orientation(imgx, imgy):
    """
    Fuction that createas gradient orientation image

    :param imgx: image gradient in the x direction
    :param imgy: image gradient in the y direction
    :return gradient oriretation image

    """
    orientation = np.arctan2(imgy, imgx) * 180 / np.pi
    return orientation

def non_maximum_suppression(img1, min_eigenvalue, orientation_img):
    """
    Function that apply the non maximum suppression

    :param img1: the image after gradient magnitude
    :param orientation_img: the image after gradient orientation
    :result the edge image 

    """
    # Create deep copy
    result = copy.deepcopy(min_eigenvalue)
    # Non-maximum suppression
    for i in range(1, img1.shape[0]-1):
        for j in range(1, img1.shape[1]-1):
            cur_direction = orientation_img[i, j]
            if cur_direction < 0:
                orientation_img[i, j] += 180
                cur_direction = orientation_img[i, j]
            if (0 <= cur_direction < 22.5) or (157.5 <= cur_direction <= 180):
                if not(img1[i, j] >= img1[i, j-1]) or not(img1[i, j] >= img1[i, j+1]):
                    result[i, j] = 0
            elif (22.5 <= cur_direction < 67.5):
                if not(img1[i, j] >= img1[i-1, j+1]) or not(img1[i, j] >= img1[i+1, j-1]):
                    result[i, j] = 0
            elif (67.5 <= cur_direction < 112.5):
                if not(img1[i, j] >= img1[i-1, j]) or not(img1[i, j] >= img1[i+1, j]):
                    result[i, j] = 0
            elif (112.5 <= cur_direction < 157.5):
                if not(img1[i, j] >= img1[i-1, j-1]) or not(img1[i, j] >=  img1[i+1, j+1]):
                    result[i, j] = 0
    return result

def apply_threshold(img1):
    """
    Function that apply a simple threshold to image

    :param img1: a grayscale image
    :return the image that has pixels above the threshold
    
    """
    threshold = 100
    img1[img1 < threshold] = 0
    img1[img1 >= threshold] = 255
    return img1

def non_maxima_algorithm(img0, min_eigenvalue, sigma):
    """
    Function that thin out the potential corners

    :param img0: a min eigen value image
    :param sigma: scalar (standard deviation of the Gaussian smoothing kernel)
    :return the thin out min eigen value image

    """
    # Smoothing image
    img0 = gaussian_blur(img0, sigma)

    # Sobel filters
    imgx, imgy = sobel_filters(img0)

    # Gradient orientation
    orientation_img = gradient_orientation(imgx, imgy)

    # Non maximum suppression
    result = non_maximum_suppression(img0, min_eigenvalue, orientation_img)

    return apply_threshold(result)