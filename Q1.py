import cv2
import numpy as np
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt


# original image
pic = cv2.imread('tiger.jpg', 0) #to convert the image to gray scale
pic = pic/255  #normalization
x, y = pic.shape #take x,y which is the total no of coloums & rows from the image

# create gaussian noise(random number) 
mean = 0
var = 0.01
sigma = np.sqrt(var)
g = np.random.normal(loc=mean, 
                     scale=sigma, 
                     size=(x,y))

# add a gaussian noise
g_pic = pic + g


# blank image
s_pic = np.zeros((x,y), dtype=np.float32)
# salt and pepper amount
pepper = 0.1 #can be variable from 0:1
salt = 1 - pepper
# create salt and peper noise image    
for i in range(x):
    for j in range(y):
        rdn = np.random.random()
        if rdn < pepper:
            s_pic[i][j] = 0
        elif rdn > salt:
            s_pic[i][j] = 1
        else:
            s_pic[i][j] = pic[i][j]

# uniform noise
a = 0
b = 0.2
u = np.zeros((x,y), dtype=np.float64)
for i in range(x):
    for j in range(y):
        u[i][j] = np.random.uniform(a,b)

# add noise to image
u_pic = pic + u
u_pic= np.clip(u_pic, 0, 1)

# display all
cv2.imshow('original image', pic)
cv2.imshow('image with gaussian noise', g_pic)
cv2.imshow('image with salt & pepper noise', s_pic)
cv2.imshow('image with uniform noise', u_pic)
cv2.waitKey(0)
cv2.destroyAllWindows()
