#!/usr/bin/env python
# coding: utf-8

# In[13]:


from PIL import Image
import numpy as np
# Creating the 144 X 144 NumPy Array with random values
arr = np.random.randint(255, size=(800, 500), dtype=np.uint8)
# Converting the numpy array into image
img  = Image.fromarray(arr)
# Saving the image
img.save("Image_from_array.png")
print(" The Image is saved successfully")
img.show()


# In[15]:


#displaying current working img

import os
os.getcwd()


# In[16]:


image1 = Image.open("earth.jpg")

image1


# In[17]:


image1.show()


# In[19]:


image1.save('100.png')


# In[20]:


os.listdir()


# In[22]:


#creating folder
os.mkdir('sandhya')


# In[32]:


#storing an images to pdf
for f in os.listdir("."):
    if f.endswith(".jpg"):
        i = Image.open(f)
        fn, fext = os.path.splitext(f)
        i.save("sandhya/{}.pdf".format(fn))


# In[28]:


#displaying jpg images
for f in os.listdir("."):
    if f.endswith(".jpg"):
        i = Image.open(f)
        fn, fext = os.path.splitext(f)
        print(fn, "&", fext)


# In[29]:


#displaying jpg images
for f in os.listdir("."):
    if f.endswith(".jpg"):
        print(f)


# In[33]:


# Creating new multiple Directories using OS library
os.makedirs('resize//small')
os.makedirs('resize//tiny')


# In[34]:


#We can resize the images in the size as well as the aspect ratio we want. We will make one more folder inside the base folder we are working on, called “resize” and inside that folder, we will have two more folders. I want to have photos for two different purposes, one for posting as postcards on the webpage, and another for smaller thumbnails of those postcards. So I will save them in separate folders. Let’s name the folders “small” and “tiny”. .thumbnail() is the method used to change the size here, and the argument is a tuple of the size we want.
size_small = (600,600) # small images of 600 X 600 pixels
size_tiny = (200,200)  # tiny images of 200 X 200 pixels
for f in os.listdir("."):
    if f.endswith(".jpg"):
        i = Image.open(f)
        fn, fext = os.path.splitext(f)
        i.thumbnail(size_small)
        i.save("resize/small/{}_small{}".format(fn, fext))
        i.thumbnail(size_tiny)
        i.save("resize/tiny/{}_tiny{}".format(fn, fext))


# In[36]:


os.mkdir('rotate')


# In[37]:


for f in os.listdir("."):
    if f.endswith(".jpg"):
        i = Image.open(f)
        fn, fext = os.path.splitext(f)
        im = i.rotate(90)
        im.save("rotate/{}_rot.{}".format(fn, fext))


# In[ ]:




