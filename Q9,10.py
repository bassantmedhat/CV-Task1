import flask
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cmath
import math
app = flask(__name__, template_folder="templates")


def prepare(path):
    image = cv2.imread(path)
    image= cv2.cvtColor( image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize( image, (1280, 720))

    return image


def fft(img):
   rows = len(img)
   if rows == 1:
      return img

   Feven, Fodd = fft(img[0::2]), fft(img[1::2])

   combined = [0] * rows
   for i in range(int(rows/2)):
     expTerm = cmath.exp((-2.0 * cmath.pi * 1j * i) / rows)
     combined[i] = Feven[i] + (expTerm * Fodd[i])
     combined[i + int(rows/2)] = Feven[i] - expTerm * Fodd[i]
   return combined

def pad2(img):
   rows, cols = np.shape(img)
   M, N = 2 ** int(math.ceil(math.log(rows, 2))), 2 ** int(math.ceil(math.log(cols, 2)))
   F = np.zeros((M,N), dtype = img.dtype)
   F[0:rows, 0:cols] = img
   return F 

def fft2(img):
   img = pad2(img)
   return np.transpose(fft(np.transpose(fft(img))))

def ifft2(img,rows,cols):
   img = fft2(np.conj(img))
   img = np.matrix(np.real(np.conj(img)))/(rows*cols)
   return img[0:rows, 0:cols]


def fftshift(img):
   rows, cols = img.shape
   firstQuad, secondQuad = img[0: int(rows/2), 0: int(cols/2)], img[int(rows/2): rows, 0: int(cols/2)]
   thirdQuad, fourthQuad = img[0: int(rows/2), int(cols/2): cols], img[int(rows/2): rows, int(cols/2): cols]
   shiftedImg = np.zeros(img.shape,dtype = img.dtype)
   shiftedImg[int(rows/2): rows, int(cols/2): cols], shiftedImg[0: int(rows/2), 0: int(cols/2)] = firstQuad, fourthQuad
   shiftedImg[int(rows/2): rows, 0: int(cols/2)], shiftedImg[0: int(rows/2), int(cols/2): cols]= thirdQuad, secondQuad
   return shiftedImg


def highPassFiltering(img,size):#Transfer parameters are Fourier transform spectrogram and filter size
    h, w = img.shape[0:2]#Getting image properties
    h1,w1 = int(h/2), int(w/2)#Find the center point of the Fourier spectrum
    img[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 0#Center point plus or minus half of the filter size, forming a filter size that defines the size, then set to 0
    return img


def lowPassFiltering(img,size):#Transfer parameters are Fourier transform spectrogram and filter size
    h, w = img.shape[0:2]#Getting image properties
    h1,w1 = int(h/2), int(w/2)#Find the center point of the Fourier spectrum
    img2 = np.zeros((h, w), np.uint8)#Define a blank black image with the same size as the Fourier Transform Transfer
    img2[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 1#Center point plus or minus half of the filter size, forming a filter size that defines the size, then set to 1, preserving the low frequency part
    img3=img2*img #A low-pass filter is obtained by multiplying the defined low-pass filter with the incoming Fourier spectrogram one-to-one.
    return img3


def image_after_highpassfilter(gray_image):

    img_dft = fft2(gray_image)
    dft_shift = fftshift(img_dft)  # Move frequency domain from upper left to middle
    #High pass filter
    dft_shift=highPassFiltering(dft_shift,200)
    # Inverse Fourier Transform

    rows,cols=gray_image.shape
    ifimg = ifft2(dft_shift,rows,cols)  # Fourier library function call
    ifimg1 = np.abs(ifimg)
    cv2.imwrite('img/image1.png',np.int8(ifimg1))
    return np.int8(ifimg1)


def image_after_lowpassfilter(gray_image):
    

    img_dft = fft2(gray_image)
    dft_shift =fftshift(img_dft)  # Move frequency domain from upper left to middle
    # Low-pass filter
    dft_shift = lowPassFiltering(dft_shift, 200)
    # Inverse Fourier Transform

    rows,cols=gray_image.shape
    ifimg = ifft2(dft_shift,rows,cols)  # Fourier library function call
    ifimg = np.abs(ifimg)
    cv2.imwrite('img/image2.png',np.int8(ifimg))
    return np.int8(ifimg)


def hybrid_images(path1,path2):
    image1 =prepare(path1)
    image2 = prepare(path2)
    image1=image_after_lowpassfilter(image1)
    image2=image_after_highpassfilter(image2)
    new_img =  image1 + image2

    # Save the image 
    cv2.imwrite('img/hybrid_image.png',new_img)
    return new_img




