
# coding: utf-8

# ### KNN (K-Nearest Neighbors Classification)

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')

# Seaborn is a nice package for plotting, but you have to use pip to install
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier


# #### Load in the Wisconsin breast cancer dataset. The attributes below will be the columns of the dataset.
# 
# Attribute Information: (class attribute has been moved to last column)
# 
#       Attribute                     Values
#    -- -----------------------------------------
#    1. Sample code number            id number
#    2. Clump Thickness               1 - 10
#    3. Uniformity of Cell Size       1 - 10
#    4. Uniformity of Cell Shape      1 - 10
#    5. Marginal Adhesion             1 - 10
#    6. Single Epithelial Cell Size   1 - 10
#    7. Bare Nuclei                   1 - 10
#    8. Bland Chromatin               1 - 10
#    9. Normal Nucleoli               1 - 10
#    10. Mitoses                       1 - 10
#    11. Class:                        (2 for benign, 4 for malignant)

# The column names are taken from the dataset info file. Create an array
# with the column names and assign them as the header when loading the
# csv.

# In[2]:

# TODO


# The class field is coded as "2" for benign and "4" as malignant. 
# - Let's recode this to a binary variable for classification, with "1" as malign and "0" as benign.

# In[4]:

# TODO


# Look at the data using seaborn's "pairplot" function. First put the dataset into "long" format, which seaborn requires for it's plotting functions:

# In[5]:

# TODO


# It's very useful and recommended to look at the correlation matrix:

# In[6]:

# TODO


# Most of these predictors are highly correlated with the "class" variable. This is already an indication that our classifier is very likely to perform well.
# 
# We can plot out in detail how the classes distribute across the variables using the very useful pairplot() function from seaborn:

# In[7]:

# TODO


# Let's see how the kNN classifier performs on the dataset (using cross-validation).
# 
# We are going to set some parameters in the classifier constructor. Some clarification below:
# 
# 1. **n_neighbors** specifies how many neighbors will vote on the class
# 2. **weights** uniform weights indicate that all neighbors have the same weight
# 3. **metric** and **p** when distance is minkowski (the default) and p == 2 (the default), this is equivalent to the euclidean distance metric
# 
# Also load scikit's handy cross-validation module and perform the crossval

# In[8]:

# TODO


# In[9]:

# TODO


# - As you can see the accuracy is very high with 5 neighbors. [NOTE: ask what might be wrong with accuracy as a metric here].
# 
# - Let's see what it's like when we use only 1 neighbor:

# In[10]:

# TODO


# - Even with 1 neighbor we do quite well at predicting the malignant observations.
# 
# - Now fit a kNN classifier with n_neighbors=5 using just 'clump thickness' and 'cell size uniformity' as variables.
# 
# - Plot the points and the colors of where the classifier decides between malignant vs. benign. The size of the circles denotes the number of observations at that point. The hue of the circle denotes the mixture of class labels between 0 and 1.

# In[11]:

# TODO


# In[ ]:



