import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from PIL import Image


def gaussian_filter(img, D0=20):
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    x, y = img.shape
    H = np.zeros((x, y), dtype=np.float32)
    for i in range(x):
        for j in range(y):
            D = np.sqrt((i-x/2)**2 + (j-y/2)**2)
            H[i,j] = np.exp(-D**2/(2*D0*D0)) 
    Gshift = Fshift * H
    G = np.fft.ifftshift(Gshift)
    g_filter = np.abs(np.fft.ifft2(G))
    return g_filter


def average_filter(img):
    mask = np.ones([3, 3], dtype=int)
    mask = mask / 9
    x, y = img.shape
    img_new = np.zeros([x, y])
    for i in range(1, x-1):
        for j in range(1, y-1):
            temp = img[i-1, j-1]*mask[0, 0] + img[i-1, j]*mask[0, 1] + img[i-1, j + 1]*mask[0, 2] + img[i, j-1]*mask[1, 0] + img[i, j]*mask[1, 1] + img[i, j + 1]*mask[1, 2] + img[i + 1, j-1]*mask[2, 0] + img[i + 1, j]*mask[2, 1] + img[i + 1, j + 1]*mask[2, 2]
            img_new[i, j] = temp
    img_new = img_new.astype(np.uint8)
    return img_new


def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):
        for j in range(len(data[0])):
            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])
            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final


img = cv2.imread('noisy_image.png', 0)

g_filter = gaussian_filter(img)
plt.figure(figsize=(5,5))
plt.imshow(g_filter, cmap='gray')
plt.axis('off')
plt.show()

avg_filter = average_filter(img)
cv2.imshow('Average filter', avg_filter)
cv2.waitKey(0)
cv2.destroyAllWindows()

pic = Image.open("noisy_image.png").convert("L")
arr = np.array(pic)
removed_noise = median_filter(arr, 3)
plt.figure(figsize=(5,5))
plt.imshow(removed_noise, cmap='gray')
plt.axis('off')
plt.show()

