import cv2 as cv
import numpy as np
import copy
from non_maximal_suppression import *

HARRIS_WINDOW = 'My Harris corner detector'

def minimum_eigenvalue(src_gray, block_size, aperture_size):
    """
    Function that find the minimum eigenvalue

    :param src_gray: gray source image
    :param block_size: neighborhood size
    :param aperture_size: aperture parameter for the sobel operator
    :return matrix containing minimum eigenvalues

    """
    # Find min eigenvalvalue
    return cv.cornerMinEigenVal(src_gray, block_size, aperture_size)

def fill_corners(harris):
    """
    Function that fill and display the corners

    :param harris: images containing all corners detected by harris algorithm
    :return none
    
    """
    # Create a copy of src image
    src_copy = np.copy(src)
    # Fill corners by circles
    for i in range(src_copy.shape[0]):
        for j in range(src_copy.shape[1]):
            if harris[i,j] == 255:
                cv.circle(src_copy, (j,i), 3, (0, 165, 255), cv.FILLED)
    # Displaying the image with corners
    cv.imshow(HARRIS_WINDOW, src_copy)

def thresholding(val):
    """
    Function that apply threshold to minimum eigenvalue and non maxima algorithm to thin out the corners, then display the corners detected

    :param val: val of threhold in decimal
    :return none

    """
    # Find min eigenvalue
    harris_copy = np.copy(min_eigenvalue)
    # Calculate threshold
    threshold = val / 1000
    # Applying threshold
    for i in range(src_gray.shape[0]):
        for j in range(src_gray.shape[1]):
            if min_eigenvalue[i,j] > threshold:
                harris_copy[i, j] = 255
            else:
                harris_copy[i, j] = 0
    # Applying non maxima algorithm 
    result = non_maxima_algorithm(src_gray, harris_copy, 2)
    # Fill the corners
    fill_corners(result)
    
if __name__ == "__main__":
    # Load source image and convert it to gray
    src = cv.imread('./Input_Image/box_in_scene.png')
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # Set some parameters
    block_size = 3
    aperture_size = 3
    # Find minimum eigenvalue
    min_eigenvalue = cv.cornerMinEigenVal(src_gray, block_size, aperture_size) 
    # Create Window and Trackbar
    cv.namedWindow(HARRIS_WINDOW)
    cv.createTrackbar('Threshold', HARRIS_WINDOW, 10, 100, thresholding)
    # Thresholding minimum eigenvalue
    thresholding(10)
    cv.waitKey()
