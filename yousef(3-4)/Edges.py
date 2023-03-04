import numpy as np
import cv2
from scipy.signal import convolve2d
from skimage.feature import peak_local_max


def convolution(image,kernel):
    """
    Perform 2D convolution of an image with a given kernel by flipping the kernel and then using the scipy.signal.convolve2d function.
    Clip the resulting image values to 0-255 and convert to unsigned 8-bit integers.
    
    Parameters:
        image (numpy.ndarray): 2D grayscale image array
        kernel (numpy.ndarray): 2D kernel array
    
    Returns:
        numpy.ndarray: 2D array of unsigned 8-bit integers representing the convolved image
    """
    conv_im = convolve2d(image ,kernel[::-1, ::-1]).clip(0, 255)
    conv_im = conv_im.astype(np.uint8)

    return conv_im


def zero_padding (image):
    """
    Add zero padding of 1 pixel to the top, bottom, left, and right of the image by creating a new array of zeros with dimensions 2 pixels larger than the image,
    and copying the original image into the center of the new array.
    
    Parameters:
        image (numpy.ndarray): 2D grayscale image array
    
    Returns:
        numpy.ndarray: 2D array of unsigned 8-bit integers representing the padded image
    """
    image_dimensions = image.shape
    padded_image = np.zeros((image_dimensions[0]+2, image_dimensions[1] + 2),dtype=np.uint8)
    
    for i in range(image_dimensions[0]):
        for j in range (image_dimensions[1]):
            padded_image[i+1, j+1] = image[i,j]
    
    return padded_image


def non_maximum_suppression(gradient_direction, gradient_magnitude, image, nms_thresholding = 0.1 ):
    image = np.zeros_like(gradient_magnitude)
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            angle = gradient_direction[i, j]
            if (angle < 0):
                angle += 180
            if (angle < 22.5 or angle > 157.5):
                if (gradient_magnitude[i, j] > gradient_magnitude[i, j-1] and gradient_magnitude[i, j] > gradient_magnitude[i, j+1]):
                    image[i, j] = gradient_magnitude[i, j]
            elif (22.5 <= angle < 67.5):
                if (gradient_magnitude[i, j] > gradient_magnitude[i-1, j-1] and gradient_magnitude[i, j] > gradient_magnitude[i+1, j+1]):
                    image[i, j] = gradient_magnitude[i, j]
            elif (67.5 <= angle < 112.5):
                if (gradient_magnitude[i, j] > gradient_magnitude[i-1, j] and gradient_magnitude[i, j] > gradient_magnitude[i+1, j]):
                    image[i, j] = gradient_magnitude[i, j]
            elif (112.5 <= angle < 157.5):
                if (gradient_magnitude[i, j] > gradient_magnitude[i-1, j+1] and gradient_magnitude[i, j] > gradient_magnitude[i+1, j-1]):
                    image[i, j] = gradient_magnitude[i, j]

    # perform thresholding
    image = image / np.max(image)
    
    image[image < nms_thresholding] = 0
    image[image >= nms_thresholding] = 1
    image = image * 255
    return image


def double_threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)


def hysteresis(img, weak, strong):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img



def edge_detection(image_path, detector='canny'):
    """
    Apply edge detection on the image located at `image_path` using the specified detector.

    Parameters:
    -----------
    image_path: str
        The path to the image file to be processed.
    detector: str, optional
        The type of edge detector to be used. Supported values are 'canny' (default), 'sobel', 'roberts', and 'prewitt'.

    Returns:
    --------
    None

    Saves the resulting edge detection image to a file with a name that reflects the detector used.
    """
    image = cv2.imread(image_path, 0)
    padded_image = zero_padding(image)
    mean_kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
    x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    smoothed_image = convolution(padded_image, mean_kernel)
    
    if detector == 'canny':
        x_edges = convolution(smoothed_image, x_kernel)
        y_edges = convolution(smoothed_image, y_kernel)
        gradient_magnitude = np.hypot(x_edges, y_edges)
        gradient_direction = np.degrees(np.arctan2(y_edges, x_edges))
        gradient_magnitude = gradient_magnitude.astype(np.float32)
        edges_image = non_maximum_suppression(gradient_direction, gradient_magnitude, image, 0.2)
        thresholded, weak, strong = double_threshold(edges_image)
        final_image = hysteresis(thresholded, weak, strong)
        cv2.imwrite("Canny.jpg", final_image)

    elif detector == 'sobel':
        x_edges = convolution(smoothed_image, x_kernel)
        y_edges = convolution(smoothed_image, y_kernel)
        edges_image_sobel = np.hypot(x_edges, y_edges)
        edges_image_sobel = edges_image_sobel.astype(np.uint8)
        cv2.imwrite("Sobel.jpg", edges_image_sobel)

    elif detector == 'roberts':
        kernel_x = np.array([[1, 0], [0, -1]])
        kernel_y = np.array([[0, 1], [-1, 0]])
        horizontal_detection = convolution(smoothed_image, kernel_x)
        vertical_detection = convolution(smoothed_image, kernel_y)
        edges_image_roberts = np.hypot(horizontal_detection, vertical_detection)
        edges_image_roberts = edges_image_roberts.astype(np.uint8)
        cv2.imwrite("Roberts.jpg", edges_image_roberts)
    
    elif detector == 'prewitt':
        kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        horizontal_detection = convolution(smoothed_image, kernel_x)
        vertical_detection = convolution(smoothed_image, kernel_y)
        edges_image_prewitt = np.hypot(horizontal_detection, vertical_detection)
        edges_image_prewitt = edges_image_prewitt.astype(np.uint8)
        cv2.imwrite("Prewitt.jpg", edges_image_prewitt)
    
    else:
        raise ValueError("Invalid detector specified")


edge_detection("image.jpg" , "roberts")