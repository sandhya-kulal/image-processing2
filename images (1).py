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


# In[7]:


import matplotlib .image as image
img  = image.imread("22.jpg")
print("the shape of image:", img.shape)
print("the image as array is ")
print(img)


# In[8]:


import cv2

img  = cv2.imread("22.jpg",0)
ret, bw_img = cv2.threshold(img , 127, 225, cv2.THRESH_BINARY)
bw = cv2.threshold(img ,225,127, cv2.THRESH_BINARY)
cv2.imshow("Binary", bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[9]:


import cv2
imgi = cv2.imread("31.jpg",1)
B, G, R = cv2.split(imgi)
print(B)
print(G)
print(R)


# In[10]:


imgi = cv2.imread("31.jpg",1)
cv2.imshow('Original',imgi)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[11]:


cv2.imshow('Blue',B)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[12]:


cv2.imshow('Red',R)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[14]:


cv2.imshow('Green',G)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[15]:


import cv2
im  = cv2.imread("31.jpg")
new_im = im.resize((400, 200))
ar = 1*(img.shape[1]/img.shape[0])
print("aspect ratio:")
print(ar)


# In[21]:


from PIL import Image


x = "31.jpg"
img = Image.open(x)
hori_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
hori_flip.show()
ver_flip = img.transpose(Image.FLIP_TOP_BOTTOM)
ver_flip.show()


# In[20]:


import cv2
im = cv2.imread("31.jpg")
grey= cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
LAB = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
RGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
HSV = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
YUV = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
HLS = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
cv2.imshow("CONVERT BGR TO GREY", grey)
cv2.imshow("CONVERT BGR TO LAB", LAB)
cv2.imshow("CONVERT BGR TO RGB", RGB)
cv2.imshow("hsv", HSV)
cv2.imshow("HLS", HLS)
cv2.imshow("YUV", YUV)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


import cv2
a = cv2.imread("mintu.jpg")
b = cv2.imread("chintu.jpg")

add = a+b
sub = a-b
mul = a*b
div = a/b

cv2.imshow("add", add)
cv2.imshow("sub", sub)
cv2.imshow("mul", mul)
cv2.imshow("div", div)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




