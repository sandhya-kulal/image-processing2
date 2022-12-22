#!/usr/bin/env python
# coding: utf-8

# In[9]:


#chanel 
from PIL import Image

im = Image.open("22.jpg")
print(im.mode)
im.save("22.png")
print("image converted successfully")



# In[ ]:





# In[10]:


#text on image
from PIL import Image, ImageDraw, ImageFont

img = Image.open('chintu.jpg')
d1 = ImageDraw.Draw(img)
font = ImageFont.truetype('arial.ttf', 30)
d1.text((50, 60), "Hello, flower", fill=(255, 0, 0), font = font)
img.show()


# In[ ]:





# In[11]:


#histogram
im = Image.open("23.jpg")
pl = im.histogram()
plt.bar(range(256), pl[:256], color='r', alpha = 0.5)
plt.bar(range(256), pl[256:2*256], color='g', alpha = 0.4)
plt.bar(range(256), pl[2*256:], color='b', alpha = 0.3)


# In[14]:


#blending

image1 = Image.open("chintu.jpg")
image2 = Image.open("mintu.jpg")

alphablend1 = Image.blend(image1, image2, alpha=.4)
alphablend2 = Image.blend(image1,image2, alpha=.4)

alphablend1.show()
alphablend2.show()


# In[8]:


#slicing
import matplotlib.pyplot as plt

from skimage.io import imread, imshow
doggo = imread('23.jpg')
imshow(doggo)
fig, ax = plt.subplots(1,3, figsize=(6,4), sharey = True)
ax[0].imshow(doggo[:, 0:130])
ax[0].set_title('First split')

ax[1].imshow(doggo[:, 130:200])
ax[1].set_title('second split')

ax[2].imshow(doggo[:, 200:390])
ax[2].set_title('third split')


# In[13]:


#RGB channels
import matplotlib.pyplot as plt
ch_r, ch_g, ch_b = im.split()
plt.figure(figsize = (18,6))
plt.subplot(1,3,1);
plt.imshow(ch_r, cmap=plt.cm.Reds);plt.axis('off')
plt.subplot(1,3,2);
plt.imshow(ch_g, cmap=plt.cm.Greens);plt.axis('off')
plt.subplot(1,3,3);
plt.imshow(ch_b, cmap=plt.cm.Blues);plt.axis('off')
plt.tight_layout()
plt.show()


# In[2]:


#RGB channels
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
doggo = imread('23.jpg')
imshow(doggo)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,3, figsize=(12, 4), sharey = True)
ax[0].imshow(doggo[:,:,0], cmap = 'Reds')
ax[0].set_title('Red')
ax[1].imshow(doggo[:,:,1], cmap = 'Greens')
ax[1].set_title('Green')
ax[2].imshow(doggo[:,:,2], cmap = 'Blues')
ax[2].set_title('Blue')


# In[10]:


#mean, median, std
from PIL import Image, ImageStat
im  = Image.open("23.jpg")
stat = ImageStat.Stat(im)
print("mean is:", stat.mean)
print("median is:",stat.median)
print("std deviation:", stat.stddev)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




