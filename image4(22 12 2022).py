#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#quantization 
from PIL import Image
import PIL


im1 = Image.open("globe.jpg")

# quantize a image
im1 = im1.quantize(256)


im1.show()


# In[ ]:


#interpolation

import cv2
import numpy as np

img = cv2.imread('22.jpg')
near_img = cv2.resize(img,None, fx = 10, fy = 10, interpolation = cv2.INTER_NEAREST)
cv2.imshow('Near',near_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


import cv2
import numpy as np
img = cv2.imread('22.jpg')
bilinear_img = cv2.resize(img,None, fx = 10, fy = 10, interpolation = cv2.INTER_LINEAR)
cv2.imshow('Bilinear',bilinear_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


import cv2
import numpy as np

img = cv2.imread('22.jpg')
bicubic_img = cv2.resize(img,None, fx = 10, fy = 10, interpolation = cv2.INTER_CUBIC)
cv2.imshow('Bicubic',bicubic_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[1]:


import cv2
import matplotlib.pyplot as plt

image = cv2.imread('23.jpg')
cv2.imshow("image before pyrup", image)
image = cv2.pyrDown(image)
cv2.imshow('Downsample', image)
plt.imshow(image)
cv2.waitKey(0)  
cv2.destroyAllWindows()    


# In[ ]:




