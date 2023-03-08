import cv2
import numpy as np
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt


# original image
pic = cv2.imread('tiger.jpg', 0) #to convert the image to gray scale
pic = pic/255  #normalization
x, y = pic.shape #take x,y which is the total no of coloums & rows from the image

# create gaussian noise(random number) 
image = pic
def add_gaussian_noise(image, mean =0 , var = 0.01):
    sigma = np.sqrt(var)
    noise = np.random.normal(mean, sigma, size=image.shape)
    noisy_image = image + noise
    return noisy_image

# blank image
def add_salt_pepper_noise(image, pepper_amount = 0):
    salt_amount = 1 - pepper_amount
    noisy_image = np.copy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < pepper_amount:
                noisy_image[i][j] = 0
            elif rdn > salt_amount:
                noisy_image[i][j] = 1
    return noisy_image

# uniform noise
def add_uniform_noise(image, a = 0 , b = 0.2):
    noise = np.random.uniform(a, b, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

# display all
cv2.imshow('original image', pic)
cv2.imshow('image with gaussian noise', add_gaussian_noise(pic))
cv2.imshow('image with salt & pepper noise',  add_salt_pepper_noise(pic))
cv2.imshow('image with uniform noise', add_uniform_noise(pic))
cv2.waitKey(0)
cv2.destroyAllWindows()
