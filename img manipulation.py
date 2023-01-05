#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import imageio
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category= matplotlib.cbook.mplDeprecation)
pic = imageio.imread("31.jpg")
plt.figure(figsize = (6, 6))
plt.imshow(pic);
plt.axis('off');


# In[2]:


negetive = 255-pic
plt.figure(figsize = (6, 6))
plt.imshow(negetive)
plt.axis('off')


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')

import imageio
import numpy as np
import matplotlib.pyplot as plt

pic = imageio.imread("31.jpg")
gray = lambda rgb : np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
gray = gray(pic)

max_ = np.max(gray)

def log_transform():
    return(255/np.log(1+max_))*np.log(1+gray)
plt.figure(figsize = (5, 5))
plt.imshow(log_transform(),cmap = plt.get_cmap(name = 'gray'))
plt.axis('off')


# In[6]:


import imageio

import matplotlib.pyplot as plt
pic = imageio.imread("31.jpg")

gamma = 2.2
gamma_connection = ((pic/255)**(1/gamma))
plt.figure(figsize = (5,5))
plt.imshow(gamma_connection)
plt.axis('off')


# In[7]:


import pandas as pd
df = pd.DataFrame([['A231', 'Book', 5, 3, 150], 
                   ['M441', 'Magic Staff', 10, 7, 200]],
                   columns = ['Code', 'Name', 'Price', 'Net', 'Sales')

#Suppose this are the links that contains the imagen i want to add to the DataFrame
images = ['Link 1','Link 2'] 
                              


# In[8]:


import pandas as pd
from IPython.display import Image, HTML


# In[10]:


import pandas as pd
from IPython.display import Image, HTML

 
# list of strings
lst = ['23.jpg', '21.jpg']
 

df = pd.DataFrame(lst)
display(df)


# In[19]:


# code for displaying multiple images in one figure

#import libraries
import cv2
from matplotlib import pyplot as plt

# create figure
fig = plt.figure(figsize=(10, 7))

# setting values to rows and column variables
rows = 4
columns = 2

# reading images
Image1 = cv2.imread('22.jpg')
Image2 = cv2.imread('23.jpg')
Image3 = cv2.imread('earth.jpg')
Image4 = cv2.imread('galaxy.jpg')
negetive1 = 255-Image1
fig.add_subplot(rows, columns, 2)
plt.imshow(negetive1)
plt.axis('off')
fig.add_subplot(rows, columns, 1)
plt.imshow(Image1)
plt.axis('off')
plt.title("First")
negetive2 = 255-Image2
fig.add_subplot(rows, columns, 4)
plt.imshow(negetive2)
plt.axis('off')
fig.add_subplot(rows, columns, 3)
plt.imshow(Image2)
plt.axis('off')
plt.title("Second")
# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 5)
plt.imshow(Image3)
plt.axis('off')
plt.title("Third")
# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 7)
# showing image
plt.imshow(Image4)
plt.axis('off')
plt.title("Fourth")


# In[ ]:




