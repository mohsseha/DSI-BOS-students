
# coding: utf-8

# # PCA Lab II

# In[37]:

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from sklearn import metrics

get_ipython().magic('matplotlib inline')


# ## Step 1: Setup the Data

# After you've downloaded the data from the repository, go ahead and load it with Pandas

# In[38]:

airports = pd.read_csv('C:/Users/Pat.NOAGALLERY/Documents/data_sources/airport_operations.csv')


# In[ ]:




# ## Step 2: Explore the Data

# Next - Let's plot! You can use any plotting library of your choice, but be sure to explore all of the data.

# In[39]:

airports.head()


# ## Step 3: Define the Variables

# Next, let's define the x and y variables: Airport is going to be our "x" variable

# In[40]:


x = airports.drop(['airport'], axis = 1)
y = airports['airport']


# ## Step 4: Covariance Matrix

# Then, standardize the x variable for analysis

# In[41]:

xStand = StandardScaler().fit_transform(x)


# Next, create the covariance matrix from the standardized x-values and decompose these values to find the eigenvalues and eigenvectors

# In[42]:

covMat = np.cov(xStand.T)
eigenValues, eigenVectors = np.linalg.eig(covMat)


# # Step 5: Eigenpairs

# Then, check your eigenvalues and eigenvectors:

# In[43]:

print(eigenValues)
print(eigenVectors)


# To find the principal componants, find the eigenpairs, and sort them from highest to lowest. 

# In[44]:

eigenPairs = [(np.abs(eigenValues[i]), eigenVectors[:,i]) for i in range(len(eigenValues))]
eigenPairs.sort()
eigenPairs.reverse()


# ## Step 6: Explained Variance

# In[45]:

eigenPairs
for i in eigenPairs:
    print(i[0])


# In[ ]:




# Now, calculate the explained variance and the Cumulative explained variance

# In[46]:


tot = sum(eigenValues)
var_exp = [(i / tot)*100 for i in sorted(eigenValues, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


# In[52]:

print(cum_var_exp)


# ** What does the explained variance tell us?**: Here, we can see that 81.77% of the behavior can be explained by the first two principal components

# ## Step 7: Perform the PCA

# Instead of creating the projection matrix, we're going to use Scikit's built in function. Now that we have discovered the principal componants, we have an educated idea on how many componants to pass to the function. 

# In[ ]:




# Create a dataframe from the PCA results

# In[ ]:




# Now, create a new dataframe that uses the airport and year from the original set and join the PCA results with it to form a new set

# In[ ]:




# In[ ]:




# Next, graph the results onto the new feature space

# In[ ]:




# **What does the graph tell us?**

# ## Step 8: Cluster with K-Means

# Set up the k-means clustering analysis. Use the graph from above to derive "k"

# In[ ]:




# Compute the labels and centroids

# In[ ]:




# Compute the Silhoutte Score

# In[ ]:




# Lastly, plot the new two-dimensional data along with their cluster assignments: 

# In[ ]:




# In[ ]:



