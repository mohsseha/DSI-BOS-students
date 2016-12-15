
# coding: utf-8

# # Data Workflow Lab 2
# 
# Proceed with your analysis of the Project 3 data set. You may need to compute new columns as you proceed. Fit one or more linear models to the data, investigate model fits and outliers, use regularization when appropriate.
# 
# ### Learning objectives
# - Perform exploratory analysis
# - Generate correlation matrix of the features
# - Generate linear regression models
# - Evaluate model fit
# 
# If appropriate for your models and featuers:
# - Use [feature selection](http://scikit-learn.org/stable/modules/feature_selection.html) to pick the best features
# - Try different regularization options

# In[18]:

get_ipython().magic('matplotlib inline')

from collections import defaultdict
import datetime
import time

from matplotlib import pyplot as plt
# Make the plots bigger
plt.rcParams['figure.figsize'] = 10, 10
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn import linear_model


# In[19]:

# Load the data from the previous lab
# If you changed the name you'll need to edit the next line
sales = pd.read_csv("sales.csv")

# Convert dates
time.strptime(sales["First Date"][0].replace('-',''),"%Y %m %d")


# ## Exploratory Analysis
# Make some plots, look at correlations, etc.

# In[ ]:

# Compute correlations


# In[ ]:

# Perform some exploratory analysis, make a few plots


# In[ ]:

# Fit a linear model

# Plot the data and the best fit line

# Compute the model fit


# In[ ]:

# Predict Total 2016 sales, compare to 2015


# In[ ]:

# Try per zip code or city to get better resolution



# In[ ]:

# Filter out stores that opened or closed throughout the year
# If this wasn't done already


# In[ ]:

# Fit another model

# Compute the model fit


# In[ ]:

# Predict Total 2016 sales, compare to 2015

