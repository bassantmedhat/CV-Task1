import cv2
import numpy as np
import matplotlib.pyplot as plt

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