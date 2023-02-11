import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Smoothing image
def gaussian_blur(img0, sigma):
    # hsize
    hsize = 2 * int(math.ceil(3 * sigma)) + 1
    # gaussian blur
    img0 = cv2.GaussianBlur(img0, (hsize, hsize), sigma)
    return img0

# Sobel filters
def sobel_filters(img0):
    # Sobel filter for gradient in x direction
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    imgx = cv2.filter2D(img0, cv2.CV_64F, sobel_x)
    
    # Sobel filter for gradient in y direction
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    imgy = cv2.filter2D(img0, cv2.CV_64F, sobel_y)
    
    return imgx, imgy

# Calculate gradient magnitude and orientation
def gradient_magnitude(imgx, imgy):
    img1 = np.sqrt(np.power(imgx, 2) + np.power(imgy, 2))
    return img1

# Calculate gradient magnitude and orientation
def gradient_orientation(imgx, imgy):
    orientation = np.arctan2(imgy, imgx) * 180 / np.pi
    return orientation

# Edge filter using non-maximum suppression
def my_edge_filter(img0, sigma):
    # Smoothing image
    img0 = gaussian_blur(img0, sigma)

    # Sobel filters
    imgx, imgy = sobel_filters(img0)

    # Gradient magnitude
    img1 = gradient_magnitude(imgx, imgy)

    # Gradient orientation
    orientation = gradient_orientation(imgx, imgy)

    # Non-maximum suppression
    for i in range(1, img0.shape[0]-1):
        for j in range(1, img0.shape[1]-1):
            if orientation[i, j] < 0:
                orientation[i, j] += 180
            if (0 <= orientation[i, j] < 22.5) or (157.5 <= orientation[i, j] <= 180):
                if (img1[i, j] < img1[i, j-1]) or (img1[i, j] < img1[i, j+1]):
                    img1[i, j] = 0
            elif (22.5 <= orientation[i, j] < 67.5):
                if (img1[i, j] < img1[i-1, j+1]) or (img1[i, j] < img1[i+1, j-1]):
                    img1[i, j] = 0
            elif (67.5 <= orientation[i, j] < 112.5):
                if (img1[i, j] < img1[i-1, j]) or (img1[i, j] < img1[i+1, j]):
                    img1[i, j] = 0
            elif (112.5 <= orientation[i, j] < 157.5):
                if (img1[i, j] < img1[i-1, j-1]) or (img1[i, j] < img1[i+1, j+1]):
                    img1[i, j] = 0
    
    return img1

# Edge filter with threshold 
def my_edge_filter_with_threshold(img1):
    # Thresholding
    threshold = np.max(img1) * 0.1
    img1[img1 < threshold] = 0
    return img1

if __name__ == "__main__":
    # Import image
    img = cv2.imread("./cat2.jpg", 0)
    # # Gaussian blur image
    # gaussian_blur_img = gaussian_blur(img, sigma=0)
    
    # # Sobels
    # imgx, imgy = sobel_filters(gaussian_blur_img)

    # # Gradient magnitude image
    # gradient_magnitude_img = gradient_magnitude(imgx, imgy)

    # # Gradient orientation image 
    # gradient_orientation_img = gradient_orientation(imgx, imgy)
    
    # Non_maximum image
    non_maximum_img = my_edge_filter(img, sigma=2)
    
    # Save images
    # gradient_magnitude_img = np.uint8(gradient_magnitude_img)
    # gradient_magnitude_img = cv2.normalize(gradient_magnitude_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # cv2.imwrite('Gradient_Magnitude.jpg', gradient_magnitude_img) 
    # cv2.waitKey(0)

    # gradient_orientation_img = cv2.convertScaleAbs(gradient_orientation_img)
    # cv2.imwrite('Gradient_Orientation.jpg', gradient_orientation_img)
    # cv2.waitKey(0)
    
    # non_maximum_img = cv2.cvtColor(non_maximum_img, cv2.COLOR_BGR2GRAY)
    # non_maximum_img = cv2.normalize(non_maximum_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow('Non_Maximum Suppression', non_maximum_img)
    cv2.waitKey(0)
    