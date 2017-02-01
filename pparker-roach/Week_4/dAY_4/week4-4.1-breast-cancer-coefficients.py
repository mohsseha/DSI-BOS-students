
# coding: utf-8

# In[12]:

import numpy as np
import pandas as pd

#Read in Breast Cancer Dataset
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header=None)

#Obtain the first 10 features, which are the mean figures in the dataset
data_df   = df[list(df.columns[2:12])]
target_df = df[1]

#Convert DataFrame to Numpy Array
data = data_df.as_matrix(columns=None)

#Perform Pearson correlation coefficients using Numpy
feature_correlation_matrix = np.corrcoef(data.T)
feature_correlation_matrix


# In[13]:

import seaborn as sb
get_ipython().magic('matplotlib inline')

sb.heatmap(feature_correlation_matrix)


# In[14]:

sb.corrplot(data)


# In[15]:

get_ipython().magic('pinfo sb.corrplot')


# In[ ]:



