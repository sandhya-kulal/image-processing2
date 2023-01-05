#!/usr/bin/env python
# coding: utf-8

# In[8]:


from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
my_image = Image.open('earth.jpg')
sharp = my_image.filter(ImageFilter.SHARPEN)
sharp.save('D:/sharp.jpg')
sharp.show()
plt.imshow(sharp)
plt.show()


# In[7]:


import matplotlib.pyplot as plt
img = Image.open('31.jpg')
plt.imshow(img)
plt.show()

flip = img.transpose(Image.FLIP_LEFT_RIGHT)
flip.save('D:/flip.jpg')
plt.imshow(flip)
plt.show()


# In[44]:


from PIL import Image
import matplotlib.pyplot as plt
im= Image.open('31.jpg')
width, height = im.size
im1 = im.crop((20, 30, 140, 125))
im1.show()
plt.imshow(im1)
plt.show()


# In[61]:




import pandas as pd
from IPython.core.display import display,HTML

df = pd.DataFrame(['31.jpg', '23.jpg'])

display(df)


# In[56]:





# In[ ]:




