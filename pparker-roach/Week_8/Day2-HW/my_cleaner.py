
# coding: utf-8

# In[ ]:

# accepts a line of text and strips out everything but A-Z, a-z, 0-9 and spaces.


# In[3]:

import re


# In[9]:

def clean_text(line):
    line=line.lower() 
    return re.sub('[^A-Za-z0-9 ]+', '', line)

