import cv2
import numpy as np
import matplotlib.pyplot as plt

def sticks_filter(img, n=5, i=8):
    # Compute gradient magnitude image
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))

    # Define the sticks filter kernels
    kernels = []
    for rotateCode in range(i):
        angle = 360.0 * rotateCode / i
        rot_mat = cv2.getRotationMatrix2D((n / 2, n / 2), angle, 1)
        kernel = np.zeros((n, n), dtype=np.float32)
        cv2.rectangle(kernel, (0, n // 2 - 1), (n - 1, n // 2), 1.0, -1)
        kernel = cv2.warpAffine(kernel, rot_mat, (n, n))
        kernels.append(kernel)

    # Perform sticks filtering
    filtered_img = np.zeros(gradient_magnitude.shape, dtype=np.float32)
    for kernel in kernels:
        filtered = cv2.filter2D(gradient_magnitude, cv2.CV_64F, kernel)
        filtered_img = np.maximum(filtered_img, filtered)

    return filtered_img

# Load an image and apply sticks filtering
img = cv2.imread("./cat2.jpg", cv2.IMREAD_GRAYSCALE)
filtered_img = sticks_filter(img)

# Show the results
# plt.imshow(filtered_img, cmap="gray")
# plt.show()
