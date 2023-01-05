#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import cv2

img = cv2.imread('4.jpg')
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# edge_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen_kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
img = cv2.filter2D(grayscale, -1, sharpen_kernel)

# Smooth out image
# blur = cv2.medianBlur(img, 3)
blur = cv2.GaussianBlur(img, (3,3), 0)

cv2.imshow('img',img)
cv2.imwrite('img.png',img)
cv2.imshow('blur',blur)
cv2.waitKey(0)


# In[23]:



import cv2
from matplotlib import pyplot as plt

img = cv2.imread('cat.png')
mask = cv2.imread('mass.png', 0)
dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

cv2.imwrite('cat_inpaint.png',dst )

cv2.imshow('image', img)
cv2.imshow('mask', mask)
cv2.imshow('restored image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[30]:


import cv2
import numpy as np


image = cv2.imread('22.jpg')
cv2.waitKey(0)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


edged = cv2.Canny(gray, 30, 200)
cv2.waitKey(0)


contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.imshow('Canny Edges After Contouring', edged)
cv2.waitKey(0)

print("Number of Contours found = " + str(len(contours)))

 
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[34]:


# import module
from PIL import Image, ImageChops
import matplotlib.pyplot as plt

img1 = Image.open("food1.jpg")
plt.imshow(img1)
plt.show()

img2 = Image.open("food.jpg")
plt.imshow(img2)
plt.show()

diff = ImageChops.difference(img1, img2)
plt.imshow(diff)
plt.show()


# In[17]:


import skimage.io
import skimage.util
import cv2

a = skimage.io.imread('22.jpg')
print(a.shape)


b = a // 2
c = a // 3
d = a // 4
m = skimage.util.montage([a, b, c, d], multichannel=True)
print(m.shape)


skimage.io.imsave('s1.jpg', m)

cv2.imshow("montage image", m)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


from tkinter import *
from PIL import Image,ImageTk

#Create an instance of tkinter frame
win = Tk()

#Set the geometry of tkinter frame
win.geometry("750x250")

#Create a canvas
canvas= Canvas(win, width= 600, height= 400)
canvas.pack()

#Load an image in the script
img= ImageTk.PhotoImage(Image.open("3.jpg"))

#Add image to the Canvas Items
canvas.create_image(10,10,anchor=NW,image=img)

win.mainloop()


# In[ ]:




