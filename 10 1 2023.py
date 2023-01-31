#!/usr/bin/env python
# coding: utf-8

# In[7]:



import os
from os import listdir


folder_dir = "C:/Users/User/Downloads/images"
for images in os.listdir(folder_dir):


    if (images.endswith(".png")):
        print(images)


# In[8]:


folder_dir = "C:/Users/User/Downloads/images"
for images in os.listdir(folder_dir):


#if (images.endswith(".png")):
    print(images)


# In[ ]:



        
      

   


# In[11]:


import cv2 
import os 
import glob 
import matplotlib.pyplot as plt 
from PIL import Image, ImageChops
 
#Set the path where images are stored 
img_dir = "C:/Users/User/Downloads/images"   
data_path = os.path.join(img_dir,"*") 
files = glob.glob(data_path) 
data = [] 
for f1 in files: 
    img = cv2.imread(f1,0) 
    data.append(img) 
    
    plt.figure() 
    plt.imshow(img) 
    plt.axis("off")


# In[12]:


import numpy as np
import matplotlib as plt
import pandas as pd
plt.rcParams['figure.figsize'] = (10, 8)


# In[13]:


def show_image(image,title="Image",cmap_type="gray"):
    plt.imshow(image,cmap=cmap_type)
    plt.title(title)
    plt.axis("off")
def plot_comparison(img_original,img_filtered,img_title_filtered):
    fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(10,8),sharex=True,sharey=True)
    ax1.set_title("Original")
    ax1.axis("Off")
    ax2.imshow(img_filtered,cmap=plt.cm.gray)
    ax2.set_title(img_title_filtered)
    ax2.axis("Off")


# In[14]:


from skimage.restoration import inpaint
from skimage.transform import resize
from skimage import color


# In[ ]:





# In[15]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.restoration import inpaint
from skimage.transform import resize
from skimage import color

image_with_logo = plt.imread('logo1.jpg')

# Initialize the mask
mask = np.zeros(image_with_logo.shape[:-1])

# Set the pixels where the logo is to 1
mask[110:150, 190:240] = 1

# Apply inpainting to remove the logo
image_logo_removed = inpaint.inpaint_biharmonic(image_with_logo,
                                                mask,
                                                multichannel=True)

# Show the original and logo removed images
plt.title("image_with_logo")
plt.imshow(image_with_logo)
plt.show()
plt.title("image_without_logo")
plt.imshow(image_logo_removed)
plt.show()


# In[ ]:





# In[10]:


from PIL import Image
from PIL import ImageFilter
import os

def main():
# path of the folder containing the raw images
    inPath ="D:/pics"

# path of the folder that will contain the modified image
    outPath ="D:/pics1"

    for imagePath in os.listdir(inPath):
# imagePath contains name of the image
        inputPath = os.path.join(inPath, imagePath)

# inputPath contains the full directory name
        img = Image.open(inputPath)

        fullOutPath = os.path.join(outPath, 'invert_'+imagePath)
# fullOutPath contains the path of the output
# image that needs to be generated
        img.rotate(90).save(fullOutPath)

        print(fullOutPath)

# Driver Function
if __name__ == '__main__':
    main()


# In[ ]:




