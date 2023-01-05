#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2
import numpy as np
from skimage import io 
frame = cv2.cvtColor(io.imread('crop.png'), cv2.COLOR_RGB2BGR)
image = cv2.cvtColor(io.imread('22.jpg'), cv2.COLOR_RGB2BGR)
mask = 255 * np.uint8(np.all(frame == [36, 28, 237], axis=2))
contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnt = min(contours, key=cv2.contourArea)
(x, y, w, h) = cv2.boundingRect(cnt)
frame[y:y+h, x:x+w] = cv2.resize(image, (w, h))


cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[10]:


import matplotlib.pyplot as plt
import keras_ocr
pipeline = keras_ocr.pipeline.Pipeline()
#read image from the an image path (a jpg/png file or an image url)
img = keras_ocr.tools.read(pic1.jpg
                          )
# Prediction_groups is a list of (word, box) tuples
prediction_groups = pipeline.recognize([img])
#print image with annotation and boxes
keras_ocr.tools.drawAnnotations(image=img, predictions=prediction_groups[0])


# In[20]:


import cv2
from PIL import Image
img = cv2.imread("pic1.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('vary.jpg',img_rgb)


# In[66]:


import cv2
import numpy as np

# read input
img = cv2.imread('7.jpg')

# convert to gray
gray = cv2.cvtColor(img)
#gray = cv2.cvtColor(img, cv2.COLOR_ORIGIN )
# threshold and invert
thresh = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)[1]

# apply morphology close
kernel = np.ones((3,3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)


#mask = np.zeros_like(gray, dtype=np.uint8)
#cntrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
#for c in cntrs:
   # area = cv2.contourArea(c)
    #if area < 1000:
        #cv2.drawContours(mask,[c],0,255,-1)

# do inpainting
#result1 = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
#result2 = cv2.inpaint(img,mask,3,cv2.INPAINT_NS)


cv2.imwrite('circle_text_threshold.png', thresh)
#cv2.imwrite('circle_text_mask.png', mask)
#cv2.imwrite('circle_text_inpainted_telea.png', result1)
#cv2.imwrite('circle_text_inpainted_ns.png', result2)

# show results
cv2.imshow('thresh',thresh)
#cv2.imshow('mask',mask)
#cv2.imshow('result1',result1)
#cv2.imshow('result2',result2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




