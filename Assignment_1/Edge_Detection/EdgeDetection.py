import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def myEdgeFilter(img0, sigma):
    # Gaussian smoothing
    hsize = 2 * int(math.ceil(3 * sigma)) + 1
    img0 = cv2.GaussianBlur(img0, (hsize, hsize), sigma)
    
    # Sobel filter for gradient in x direction
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    imgx = cv2.filter2D(img0, -1, sobel_x)
    
    # Sobel filter for gradient in y direction
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    imgy = cv2.filter2D(img0, -1, sobel_y)
    
    # Calculate gradient magnitude and orientation
    img1 = np.sqrt(np.power(imgx, 2) + np.power(imgy, 2))
    orientation = np.arctan2(imgy, imgx) * 180 / np.pi
    
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
    
    # Thresholding
    threshold = np.max(img1) * 0.1
    img1[img1 < threshold] = 0
    
    return img1


img = cv2.imread("./cat2.jpg", 0)
img1 = myEdgeFilter(img, sigma=1)

plt.imshow(img1, cmap="gray")
plt.show()
