#!/usr/bin/env python
# coding: utf-8

# In[9]:


#logical operation
import cv2 as cv
img1 = cv.imread('chintu.jpg')
img2 = cv.imread('mintu.jpg')
bitwise_AND = cv.bitwise_and(img1, img2)
bitwise_OR = cv.bitwise_or(img1, img2)
bitwise_NOT = cv.bitwise_not(img1)
cv.imshow('img1',img1)
cv.imshow('img2',img2)
cv.imshow('AND',bitwise_AND)
cv.imshow('OR',bitwise_OR)
cv.imshow('NOT',bitwise_NOT)
if cv.waitKey(0) & 0xff == 27: 
    cv.destroyAllWindows()


# In[10]:





# In[30]:


#median filtering 
import cv2
import cv2
import numpy as np

img_noisy1=cv2.imread("22.jpg",0)
m,n=img_noisy1.shape
img_new1=np.zeros([m,n])

for i in range(1,m-1):
    for j in range(1,n-1):
        temp=[img_noisy1[i-1,j-1],img_noisy1[i-1,j],img_noisy1[i-1,j-1],img_noisy1[i-1,j+1],img_noisy1[i,j-1],img_noisy1[i,j],img_noisy1[i,j+1],img_noisy1[i+1,j-1],img_noisy1[i+1,j],img_noisy1[i+1,j+1]]
        temp=sorted(temp)
        img_new1[i,j]=temp[4]
        img_new1=img_new1.astype(np.uint8)

cv2.imshow("MEDIAN FILTERED IMAGE",img_new1)
cv2.waitKey(0)  
cv2.destroyAllWindows()      


# In[27]:


#paste an image
from PIL import Image, ImageDraw, ImageFilter
im1 = Image.open('galaxy.jpg')
im2 = Image.open('globe.jpg')
mask = Image.new("L", im2.size, 0)
draw = ImageDraw.Draw(mask)
draw.ellipse((200, 100, 640, 490), fill = 265)
mask_blur = mask.filter(ImageFilter.GaussianBlur(10))
back = im1.copy()
back.paste(im2, (0,0), mask_blur)
back.show()


# In[31]:


#avg filtering

import cv2
import numpy as np
img=cv2.imread("22.jpg",0)
m,n=img.shape
mask=np.ones([3,3], dtype = int)
mask = mask/9
img_new = np.zeros([m, n])
for i in range(1, m-1):
    for j in range(1, n-1):
        temp = img [i-1,j-1]*mask[0,0]+img[i-1,j]*mask[0,1]+img[i-1,j+1]*mask[0,2]+img[i,j-1]*mask[1,0]+img[i,j]*mask[1,1]+img[i,j+1]*mask[1,2]+img[i+1,j-1]*mask[2,0]+img[i+1,j]*mask[2,1]+img[i+1,j+1]*mask[2,2]
        img_new[i,j] = temp
        img_new = img_new.astype(np.uint8)
cv2.imwrite('burred1.jpg', img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()  


# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


#Upsample
import cv2

image = cv2.imread('23.jpg')
cv2.imshow("image before pyrup", image)
image = cv2.pyrUp(image)
cv2.imshow('Upssample', image)
cv2.waitKey(0)  
cv2.destroyAllWindows()    


# In[5]:


#dowmsampling
import cv2

image = cv2.imread('23.jpg')
cv2.imshow("image before pyrup", image)
image = cv2.pyrDown(image)
cv2.imshow('Downsample', image)
cv2.waitKey(0)  
cv2.destroyAllWindows()    


# In[ ]:




