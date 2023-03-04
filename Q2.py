import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from PIL import Image

# open the image 
img = cv2.imread('noisy_image.png',0)
# plt.figure(figsize=(2,2))
# #plt.imshow(img, cmap='gray')
# plt.axis('off')
# # plt.show()


# transform the image into frequency domain, f --> F
F = np.fft.fft2(img)
Fshift = np.fft.fftshift(F) #shift the low freq from the corners to the middle

# Create Gaussin Filter: Low Pass Filter
x,y = img.shape
H = np.zeros((x,y), dtype=np.float32)

#D0 is the cut off freq can be any positive no
D0 = 20 #decrese the no of D0 increase the Bluring of the image
for i in range(x):
    for j in range(y):
        D = np.sqrt((i-x/2)**2 + (j-y/2)**2)  #D is the radius/distance from the center
        H[i,j] = np.exp(-D**2/(2*D0*D0))  

# Apply the gaussian filter
Gshift = Fshift * H #gaussian filter to the shift of the FT
G = np.fft.ifftshift(Gshift) #return the shift
g_filter = np.abs(np.fft.ifft2(G)) #inverse FT
plt.figure(figsize=(5,5))
plt.imshow(g_filter, cmap='gray')
plt.axis('off')
plt.show()






# average filter 
# Develop Averaging filter(3, 3) mask
mask = np.ones([3, 3], dtype = int) #filter = matrix of ones 
mask = mask / 9 #divide by the sum
   
# Convolve the 3X3 mask over the image 
img_new = np.zeros([x, y]) #x , y shape of the image
  
for i in range(1, x-1): #the corners don't include
    for j in range(1, y-1):
        temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i-1, j + 1]*mask[0, 2]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]+img[i, j + 1]*mask[1, 2]+img[i + 1, j-1]*mask[2, 0]+img[i + 1, j]*mask[2, 1]+img[i + 1, j + 1]*mask[2, 2]
         
        img_new[i, j]= temp
          
img_new = img_new.astype(np.uint8)

# m = 5 #no of rows
# n = 5 #no of columns
# avg_filter = cv2.blur(img, (m,n))

cv2.imshow('Average filter', img_new)

cv2.waitKey(0)
cv2.destroyAllWindows()




# median filter (more effecient for salt and pepper noise)
def median_filter(data, filter_size):  #data = array of the image
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
pic = Image.open("noisy_image.png").convert("L")
arr = np.array(pic)
removed_noise = median_filter(arr, 3) 
pic = Image.fromarray(removed_noise)
pic.show()


# median_filter = cv2.medianBlur(img,5)




