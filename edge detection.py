#!/usr/bin/env python
# coding: utf-8

# In[1]:


#edge detection

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

load_img = cv2.imread('emoji.jpg')

load_img = cv2.cvtColor(load_img, cv2.COLOR_BGR2RGB)

gray_img = cv2.cvtColor(load_img, cv2.COLOR_BGR2GRAY)

edge_img =  cv2.Canny(gray_img, threshold1 = 150, threshold2 = 150)

plt.figure(figsize = (50,50))
plt.subplot(1,3,1)
plt.imshow(load_img, cmap = "gray")
plt.title('original image')
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(gray_img, cmap = "gray")
plt.title('gray scale image')
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(edge_img, cmap = "gray")
plt.title('canny edge detected  image')
plt.axis("off")
plt.show()


# In[20]:


img = cv2.imread("butterfly.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(gray,(3,3),0)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobelx = cv2.Sobel(img, cv2.CV_64F,1,0,ksize = 5)
sobely = cv2.Sobel(img, cv2.CV_64F,0,1,ksize = 5)

plt.subplot(2,2,1), plt.imshow(img,cmap = 'gray') 
plt.title('original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian, cmap= "gray")
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx, cmap = "gray") 
plt.title('sobel x'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4), plt.imshow(sobely,cmap = 'gray')
plt.title('sobel v'), plt.xticks([]), plt.yticks([])

plt.show()


# In[21]:


img = cv2.imread("girl.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gau = cv2.GaussianBlur(gray,(3,3),0)

kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prex = cv2.filter2D(img_gau, -1, kernelx)
img_prey = cv2.filter2D(img_gau, -1, kernely)

cv2.imshow("original image", img)
cv2.imshow("Prewitt x", img_prex)
cv2.imshow("Prewitt y", img_prey)
cv2.imshow("Prewitt ",img_prex+ img_prey)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


from scipy import ndimage

from matplotlib import pyplot as plt

roberts_cross_v= np.array([[1, 0],
                           [0,-1]] )

roberts_cross_h =np.array([[0, 1],
                           [-1,0]] )

img= cv2.imread("7.jpg",0).astype("float64") 
img/=255.0
vertical =ndimage.convolve( img, roberts_cross_v)
horizontal= ndimage.convolve(img, roberts_cross_h)



edged_img =np.sqrt( np.square(horizontal) + np.square(vertical))
edged_img*=255

cv2.imwrite("output2.jpg",edged_img)
cv2.imshow("output1", edged_img)

cv2.waitKey()

cv2.destroyAllWindows()


# In[ ]:




