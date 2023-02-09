#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np


image1 = cv2.imread('butterfly.png')


img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)


ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV)



cv2.imshow('Binary Threshold', thresh1)
cv2.imshow('Binary Threshold Inverted', thresh2)
cv2.imshow('Truncated Threshold', thresh3)
cv2.imshow('Set to 0', thresh4)
cv2.imshow('Set to 0 Inverted', thresh5)


if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()


# In[4]:


import cv2
import numpy as np
img = cv2.imread('butterfly.png',0)
mean_data = np.mean(img)
[r,c]=img.shape
epsilon = 0.001

diff_threshold = 100
data_one = []
data_two = []


# In[5]:


img.shape


# In[ ]:


while diff_threshold > epsilon:
    for i in range(r):
         for j in range(c):
                if img[i,j] < mean_data:
                    
                    data_one.append(img[i,j])
                else:
                    data_two.append(img[i,j])
                print(data_one)
                print(data_two)
                
                mu_one = np.mean(data_one)
                mu_two = np.mean(data_two)
                avg_mean = (mu_one + mu_two)/2
                diff_avg = (abs(mean_data-avg_mean))
                mean_data = avg_mean
                print(mean_data)
        
  


# In[6]:




import cv2
import numpy as np
img=cv2.imread("butterfly.png",0)
epsilon=0.001
diff_threshold=100
mean_data=np.mean(img)
[r,c]=img.shape

while diff_threshold>epsilon:
    data_one=[]
    data_two=[]
    for i in range(r):
        for j in range(c):
            if img[i,j]<mean_data:
                data_one.append(img[i,j])
            else:
                data_two.append(img[i,j])
            print(data_one)
            print(data_two)
            mu_one=np.mean(data_one)
            mu_two=np.mean(data_two)
            avg_mean=(mu_one+mu_two)/2
            diff_threshold=abs(mean_data-avg_mean)
            mean_data=avg_mean
            print(mean_data)


# In[ ]:





# In[ ]:


import cv2 
import numpy as np

img = cv2.imread('butterfly.png', cv2.IMREAD_GRAYSCALE)  
cv2.imshow('gray', img)
cv2.imwrite('gray1.jpg',img)

blur = cv2.GaussianBlur(img,(7,7),0)
cv2.imshow('blur', img)
cv2.imwrite('blur1.jpg',img)

x,threshold = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY) 
cv2.imshow('Binary threshold', threshold)
cv2.imwrite('binarythresh1.jpg',img)

ret2,th2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('Otsus Thresholding', th2)
cv2.imwrite('Otsus.jpg',img)

cv2.waitKey(0)


# In[ ]:




