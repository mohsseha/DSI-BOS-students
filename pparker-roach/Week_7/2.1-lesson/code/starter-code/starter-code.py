
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import StandardScaler


# In[4]:

iris = pd.read_csv('C:/Users/Pat.NOAGALLERY/Documents/data_sources/iris.csv')


# In[5]:

iris.head()


# ## Step 1: Split the set into two sets

# "X" will be the data and "Y" will be the class labels

# In[10]:

X = iris.drop(['Name'], axis = 1)
y = iris['Name']


# ## Step 2: Explore the Data

# Next - Let's plot! You can use any plotting library of your choice, but be sure to explore all of the data. 

# In[ ]:




# ## Step 3: Dimensionality Reduction

# First, standarize the data. While the Iris data attributes are all measured in the same units (cm), this is a worthwhile step for optimization and good practice for more unruly datasets!

# In[12]:




# Now, let's set up our data for decomposition by creating a covariance matrix

# Now, decompose the the covariance matrix

# In[16]:




# Check the eigenvalues and eigenvectors

# In[ ]:




# In[ ]:




# The eigenvectors with the lowest eigenvalues can be dropped

# In[ ]:




# Calculate the explained variance

# In[27]:

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


# What does the explained variance tell us?

# In[ ]:



