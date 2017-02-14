
# coding: utf-8

# <center><h1>General Assembly Data Science Immersion Program</h1>
# 
# <h1>Capstone Project</h1>
# <h1>Is This Mushroom Edible or Poisonous</h1></center>

# # Executive Summary 

# ## Problem Statement
# <>
# 
# 
# ## Goal
# <> 
#    
# ## Deliverables
#     *<>
#     

# # Summary of Findings

# tbd

# ## Supporting Graphics

# tbd

# # Recommendations for Next Steps

# tbd

# ## Data Description
# <>

# ## The Model

# tbd

# * <>
# 

# #### The following block is used to import all Python libraries used in this model

# In[80]:

import numpy as np
import pandas as pd
import matplotlib as mpl

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')


# #### The following block imports the mushroom classification data stored in mushrooms.csv

# In[81]:

shrooms = pd.read_csv("mushrooms.csv")


# #### Let's take a look at some of the features of this data

# In[82]:

shrooms.head()


# In[83]:

shrooms.shape


# In[84]:

shrooms.columns


# In[85]:

shrooms.describe()


# #### The data is encoded as an alpha character which maps to an attribute as described in the Data Description section above. Let's convert those alpha characters to dummy columns in order to start looking for correlations.
# 

# In[86]:

shrooms.columns


# In[87]:

columns = shrooms.columns
columns = columns.drop('class') # we won't make dummy columns for our target column
for col in columns:
    for data in shrooms[col].unique():
        shrooms[col + "_" + data] = shrooms[col] == data

# we can now drop the original columns
for col in columns:
    shrooms.drop(col, inplace = True, axis = 1)



# In[88]:

shrooms.head()


# In[89]:

# and now let's declare that in our target variable 'class' that p=1 (poisonous) and e=0 (edible)
# and convert the data in y accordingly

shrooms['class'] = shrooms['class'] == 'p'


# In[ ]:




# In[90]:

shrooms.head()


# In[91]:

# Now let's replace all of the bool values (True/False) with integers (1/0)
# columns = shrooms.columns
# for col in columns:
#     print(col)
#     shrooms[col] = shrooms[col].astype(int)
shrooms = shrooms.astype(int)


# In[92]:

shrooms.head()


# In[95]:

#plt.scatter()


# In[ ]:



