#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
image = cv2.imread("22.jpg")
cv2.imshow('images',image)
cv2.imwrite("D:\img.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[2]:



image = cv2.imread("22.jpg", 0)
cv2.imshow('images',image)
cv2.imwrite("D:/img.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


image = cv2.imread("C:/Users/User/Downloads/images/23.jpg")
cv2.imshow('images',image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[4]:



from PIL import Image


x = "23.jpg"
img = Image.open(x)


width = img.width
height = img.height


print("The height of the image is: ", height)
print("The width of the image is: ", width)


# In[5]:


import numpy

image = cv2.imread("22.jpg")
print("no of channels is:"+str(image.ndim))
print("no of channels is:", image.shape)
cv2.imshow('images',image)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[6]:


from PIL import Image
filepath = "31.jpg"
im= Image.open(filepath)
new_im = im.resize((300, 200))
new_im


# In[9]:


import matplotlib .image as image
img  = image.imread("22.jpg")
print("the shape of image:", img.shape)
print("the image as array is ")
print(img)


# In[38]:


import cv2

img  = cv2.imread("22.jpg",0)
ret, bw_img = cv2.threshold(img , 127, 225, cv2.THRESH_BINARY)
bw = cv2.threshold(img ,225,127, cv2.THRESH_BINARY)
cv2.imshow("Binary", bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[28]:


import cv2
imgi = cv2.imread("31.jpg",1)
B, G, R = cv2.split(imgi)
print(B)
print(G)
print(R)


# In[34]:


imgi = cv2.imread("31.jpg",1)
cv2.imshow('Original',imgi)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[35]:


cv2.imshow('Blue',B)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[36]:


cv2.imshow('Red',R)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[37]:


cv2.imshow('Green',G)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[39]:


import cv2
im  = cv2.imread("31.jpg")
new_im = im.resize((400, 200))
ar = 1*(img.shape[1]/img.shape[0])
print("aspect ratio:")
print(ar)


# In[ ]:




