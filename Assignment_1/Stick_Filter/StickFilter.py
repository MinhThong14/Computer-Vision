import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import copy

INPUT_PATH = "./Input_Image"
OUTPUT_PATH = "./Output_Image"
OUTPUT_PATH_EDGE = "../Edge_Dection/Output_Image"

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
    imgx = cv2.filter2D(img0, cv2.CV_64F, sobel_x)
    
    # Sobel filter for gradient in y direction
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    imgy = cv2.filter2D(img0, cv2.CV_64F, sobel_y)
    
    return imgx, imgy

def gradient_magnitude(imgx, imgy):
    """
    Function that creates the gradient magnitude image

    :param imgx: image gradient in the x direction
    :param imgy: image gradient in the y direction
    :return gradient magnitude image
    """
    img1 = np.sqrt(np.power(imgx, 2) + np.power(imgy, 2))
    return img1

def sticks_filter(img, n=5, i=8):
    """
    Function that create sticks filter

    :param img: image that apply the sticks filter
    :param n: kernel size
    :param i: number of sticks
    :return the image after apply the sticks  

    """
    # Define the sticks filter kernels
    kernels = np.array([
        [   [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1/n, 1/n, 1/n, 1/n, 1/n],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        [   [0, 0, 0, 0, 0],
            [0, 0, 0, 1/n, 1/n],
            [0, 0, 1/n, 0, 0],
            [1/n, 1/n, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        [   [0, 0, 0, 0, 1/n],
            [0, 0, 0, 1/n, 0],
            [0, 0, 1/n, 0, 0],
            [0, 1/n, 0, 0, 0],
            [1/n, 0, 0, 0, 0],
        ],
        [   [0, 0, 0, 1/n, 0],
            [0, 0, 0, 1/n, 0],
            [0, 0, 1/n, 0, 0],
            [0, 1/n, 0, 0, 0],
            [0, 1/n, 0, 0, 0],
        ],
        [   [0, 0, 1/n, 0, 0],
            [0, 0, 1/n, 0, 0],
            [0, 0, 1/n, 0, 0],
            [0, 0, 1/n, 0, 0],
            [0, 0, 1/n, 0, 0],
        ],
        [   [0, 1/n, 0, 0, 0],
            [0, 1/n, 0, 0, 0],
            [0, 0, 1/n, 0, 0],
            [0, 0, 0, 1/n, 0],
            [0, 0, 0, 1/n, 0],
        ],
        [   [1/n, 0, 0, 0, 0],
            [0, 1/n, 0, 0, 0],
            [0, 0, 1/n, 0, 0],
            [0, 0, 0, 1/n, 0],
            [0, 0, 0, 0, 1/n],
        ],
        [   [0, 0, 0, 0, 0],
            [1/n, 1/n, 0, 0, 0],
            [0, 0, 1/n, 0, 0],
            [0, 0, 0, 1/n, 1/n],
            [0, 0, 0, 0, 0],
        ],
        
    ])

    # Perform sticks filtering
    filtered_img = np.zeros(img.shape, dtype=np.float32)
    # print(img)
    for i in range(1, img.shape[0]-n-1, n):
        for j in range(1, img.shape[1]-n-1, n):
            max_kernel, max_contrast = kernels[0], float('-inf')
            for kernel in kernels:
                sum_sticks = 0
                sum_neighboor = 0
                for h in range(n):
                    for g in range(n):
                        if kernel[h, g] != 0:
                            sum_sticks += img[i+h, j+g]
                        else:
                            sum_neighboor += img[i+h, j+g]
                avg_sticks = math.ceil(sum_sticks / 5)
                avg_neighboor = math.ceil(sum_neighboor / 5**2)
                if avg_sticks - avg_neighboor > max_contrast:
                    max_contrast = avg_sticks - avg_neighboor
                    max_kernel = copy.deepcopy(kernel)

            filtered = cv2.filter2D(img, cv2.CV_64F, max_kernel)
            filtered_img = np.maximum(filtered_img, filtered)
    return filtered_img

def gradient_orientation(imgx, imgy):
    """
    Fuction that createas gradient orientation image

    :param imgx: image gradient in the x direction
    :param imgy: image gradient in the y direction
    :return gradient oriretation image

    """
    orientation = np.arctan2(imgy, imgx) * 180 / np.pi
    return orientation

def non_maximum_suppression(img1, orientation_img):
    """
    Function that apply the non maximum suppression

    :param img1: the image after gradient magnitude
    :param orientation_img: the image after gradient orientation
    :result the edge image 

    """
    # Create deep copy
    result = copy.deepcopy(img1)
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
    threshold = 30
    img1[img1 < threshold] = 0
    img1[img1 >= threshold] = 255
    return img1

def my_edge_filter_with_stick_filter(img0, sigma):
    """
    Function that dectect the edge of an image

    :param img0: a gray scale image
    :param sigma: scalar (standard deviation of the Gaussian smoothing kernel)
    :return the the edge magnitude image

    """
    # Smoothing image
    img0 = gaussian_blur(img0, sigma)

    # Sobel filters
    imgx, imgy = sobel_filters(img0)

    # Gradient magnitude
    img1 = gradient_magnitude(imgx, imgy)

    # Sticks filter
    img1 = sticks_filter(img1)

    # Gradient orientation
    orientation_img = gradient_orientation(imgx, imgy)

    # Non maximum suppression
    result = non_maximum_suppression(img1, orientation_img) 

    return apply_threshold(result)


if __name__ == "__main__":
    # Input images
    input_images = ["cat2.jpg", "img0.jpg", "littledog.jpg"]

    # Genetate edge dection for all input images
    for input_img in input_images:
        input = INPUT_PATH + '/' + input_img
        # Import image
        img = cv2.imread(input,0)
        # Gaussian blur image
        gaussian_blur_img = gaussian_blur(img, sigma=0)
        
        # Sobels
        imgx, imgy = sobel_filters(gaussian_blur_img)

        # Gradient magnitude image
        gradient_magnitude_img = gradient_magnitude(imgx, imgy)

        # Gradient orientation image 
        gradient_orientation_img = gradient_orientation(imgx, imgy)
        
        # Stick filter on gradient magnitude image
        sticks_filter_magnitude_img = sticks_filter(gradient_magnitude_img)

        # Non_maximum image
        non_maximum_img_with_sticks_filter = my_edge_filter_with_stick_filter(img, sigma=2)

        # Image name
        img_name = input_img.split('.')[0]

        # Save images
        sticks_filter_magnitude_img = np.uint8(sticks_filter_magnitude_img)
        sticks_filter_magnitude_img = cv2.normalize(sticks_filter_magnitude_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        output_magnitude_img_with_sticks_filter = OUTPUT_PATH + '/' + img_name + '/' + img_name + '_gradient_magnitude_with_sticks_filter.jpg'
        cv2.imwrite(output_magnitude_img_with_sticks_filter , sticks_filter_magnitude_img) 

        output_non_maximum_img_with_sticks_filter = OUTPUT_PATH + '/' + img_name + '/' + img_name + '_non_maximum_with_sticks_filter.jpg'
        cv2.imwrite(output_non_maximum_img_with_sticks_filter, non_maximum_img_with_sticks_filter)

        output_non_maximum_img_with_sticks_filter_to_edge_folder = OUTPUT_PATH_EDGE + '/' + img_name + '/' + img_name + '_non_maximum_with_sticks_filter.jpg'
        cv2.imwrite(output_non_maximum_img_with_sticks_filter_to_edge_folder, non_maximum_img_with_sticks_filter)

