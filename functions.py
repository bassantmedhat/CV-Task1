import cv2
import numpy as np
import matplotlib.pyplot as plt
import cmath
import math

from math import sqrt,exp


#################################################################################################################
def rgb_to_gray(source: np.ndarray):
    gray_image= np.dot(source[..., :3], [0.299, 0.587, 0.114]).astype('uint8')
    return gray_image
#################################################################################################################

def global_threshold1(source: np.ndarray, threshold: int):
    # # source: gray image   
    # src = np.copy(source)
    # row, column = src.shape
    # for x in range(column):
    #     for y in range(row):
    #         if src[x,y] > threshold:
    #              src[x,y] = 1
    #         else:
    #              src[x,y] = 0
    # return src
    src = np.copy(source)
    if len(src.shape) > 2:
        src = rgb_to_gray(source)
    return (src > threshold).astype('int')
#################################################################################################################
def local_threshold1(source: np.ndarray, divs: int):
    # # source: gray image 
    # Split of the image
    src = np.copy(source)
    for row in range(0,src.shape[0],divs):
        for col in range(0,src.shape[1],divs):
            mask_src=src[row:row+divs,col:col+divs]
            threshold=int(np.mean(mask_src))-10
            src[row:row+divs, col:col + divs]= global_threshold1(source=mask_src,threshold=threshold)
    return src

#################################################################################################################
def histogram(source: np.array,bins_num: int= 256):
    if bins_num == 2:
        new_data = source
    else:
        new_data =source.astype('uint8')
    bins = np.arange(0, bins_num)
    hist = np.bincount(new_data.ravel(), minlength= bins_num)
    return hist, bins

#################################################################################################################
def equalize_histogram(source: np.ndarray, bins_num : int = 256):
    #     
    bins = np.arange(0, bins_num)

    # Calculate the Occurrences of each pixel in the input
    hist_array = np.bincount(source.flatten(), minlength=bins_num)

    # Normalize Resulted array
    px_count = np.sum(hist_array)
    hist_array = hist_array / px_count

    # Calculate the Cumulative Sum
    hist_array = np.cumsum(hist_array)

    # Pixel Mapping
    trans_map = np.floor(256 * hist_array).astype('uint8')

    # Transform Mapping to Image
    img1d = list(source.flatten())
    map_img1d = [trans_map[px] for px in img1d]

    # Map Image to 2d & Reshape Image
    img_eq = np.reshape(np.asarray(map_img1d), source.shape)

    return img_eq , bins

#################################################################################################################
def normalize_histogram(source: np.ndarray, bins_num : int = 256):
    mn = np.min(source)
    mx = np.max(source)
    norm = ((source - mn) * (256 / (mx - mn))).astype('uint8')
    hist, bins = histogram(norm, bins_num=bins_num)
    return norm, hist, bins

#################################################################################################################
def draw_rgb_histogram(source: np.ndarray):

    # colors = ["red" , "green","blue"]
    colors =  [(0, 0, 1),(0, 1, 0),(1, 0, 0)]

    for i in range(source.shape[2]):
        hist, bins = histogram(source=source[:, :, i], bins_num=256)
        plt.plot(bins, hist, color=colors[i])
#################################################################################################################
def draw_gray_histogram(source: np.ndarray, bins_num):

    # Create histogram and plot it
    hist, bins = histogram (source=source, bins_num=bins_num)
    plt.plot(bins, hist)

#################################################################################################################
def display_bar_graph( x ,height, width, color):
    plt.bar(x=x, height=height, width=width, color=color)

#################################################################################################################
def rgb_distribution_curve(source: np.ndarray):
        colors = ["red" , "green","blue"]
        for i in range(source.shape[2]):
            hist, bins = histogram(source=source[:, :, i], bins_num=256)
            pdf = (250*hist) / sum( hist)
            cdf = np.cumsum(pdf)
            cdf = np.floor(255 *cdf ).astype('uint8')
            plt.plot(bins, cdf, label="CDF", color=colors[i])
    

#################################################################################################################
##################################### Tab 3 (High and Low) ########################################################
#################################################################################################################
def prepare(path):
    image = cv2.imread(path)
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (210,210))

    return image
#################################################################################################################
#################################################################################################################
def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
def idealFilterHP(D0,imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 0
    return base
def idealFilterLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 1
    return base
def image_after_highpassfilter(path):
    img =prepare(path)
    original = np.fft.fft2(img)
    center = np.fft.fftshift(original) # Move frequency domain from upper left to middle
    
    #High pass filter
    HighPass = idealFilterHP(50,img.shape)
    HighPassCenter = center * idealFilterHP(15,img.shape)
    HighPass = np.fft.ifftshift(HighPassCenter)
    # Inverse Fourier Transform
    inverse_HighPass = np.fft.ifft2(HighPass)  # Fourier library function call
    ifimg1 = np.abs(inverse_HighPass)
    cv2.imwrite('img/image1.png',np.abs(ifimg1))

    return 'img/image1.png'

#################################################################################################################
def image_after_lowpassfilter(path):
    
    gray_image =prepare(path)
    original = np.fft.fft2(gray_image)
    center = np.fft.fftshift(original)

    # Low-pass filter
    LowPass = idealFilterLP(50,gray_image.shape)
    # Inverse Fourier Transform
    LowPassCenter = center * idealFilterLP(15,gray_image.shape)
    # rows,cols=gray_image.shape
    LowPass = np.fft.ifftshift(LowPassCenter)
    inverse_LowPass = np.fft.ifft2(LowPass)
    ifimg = np.abs(inverse_LowPass)
    cv2.imwrite('img/image2.png',np.abs(ifimg))
    return 'img/image2.png'

#################################################################################################################
def getfilter(path,flag):
    if flag == 1:
        return image_after_highpassfilter(path)
    else :
         return image_after_lowpassfilter(path)


def hybrid_images(path1 ,path2):

    image1 = prepare(path1)
    image2 = prepare(path2)
    new_img =  image1 + image2

    # Save the image 
    cv2.imwrite('img/hybrid_image.png',new_img)
    return 'img/hybrid_image.png'



#################################################################################################################