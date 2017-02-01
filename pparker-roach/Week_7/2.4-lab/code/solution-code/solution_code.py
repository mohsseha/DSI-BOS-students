
# coding: utf-8

# # PCA Lab II

# In[ ]:

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from sklearn import metrics


get_ipython().magic('matplotlib inline')


# ## Step 1: Setup the Data

# After you've downloaded the data from the repository, go ahead and load it with Pandas

# In[ ]:

airports = pd.read_csv('../../assets/datasets/airport_operations.csv')


# In[ ]:

airports.head()


# ## Step 2: Explore the Data

# Next - Let's plot! You can use any plotting library of your choice, but be sure to explore all of the data.

# In[ ]:




# ## Step 3: Define the Variables

# Next, let's define the x and y variables: Airport is going to be our "x" variable

# In[ ]:

x = airports.ix[:,2:14].values
y = airports.ix[:,0].values


# ## Step 4: Covariance Matrix

# Then, standardize the x variable for analysis

# In[ ]:

xStand = StandardScaler().fit_transform(x)


# Next, create the covariance matrix from the standardized x-values and decompose these values to find the eigenvalues and eigenvectors

# In[ ]:

covMat = np.cov(xStand.T)
eigenValues, eigenVectors = np.linalg.eig(covMat)


# # Step 5: Eigenpairs

# Then, check your eigenvalues and eigenvectors:

# In[1]:

print(eigenValues)
print(eigenVectors)


# To find the principal componants, find th eigenpairs, and sort them from highest to lowest. 

# In[ ]:

eigenPairs = [(np.abs(eigenValues[i]), eigenVectors[:,i]) for i in range(len(eigenValues))]
eigenPairs.sort()
eigenPairs.reverse()
for i in eigenPairs:
    print(i[0])


# ## Step 6: Explained Variance

# In[ ]:

totalEigen = sum(eigenValues)
varExpl = [(i / totalEigen)*100 for i in sorted(eigenValues, reverse=True)]


# In[ ]:

print(varExpl)


# Now, calculate the explained variance and the Cumulative explained variance

# In[ ]:

cvarex = np.cumsum(varExpl)


# In[ ]:

print(cvarex)


# ** What does the explained variance tell us?**: Here, we can see that 81.77% of the behavior can be explained by the first two principal componants

# ## Step 7: Perform the PCA

# Instead of creating the projection matrix, we're going to use Scikit's built in function. Now that we have discovered the principal componants, we have an educated idea on how many componants to pass to the function. 

# In[ ]:

pcask = PCA(n_components=2)
Y = pcask.fit_transform(xStand)


# Create a dataframe from the PCA results

# In[ ]:

Ydf = pd.DataFrame(Y, columns=["PC1", "PC2"])


# Now, create a new dataframe that uses the airport and year from the original set and join the PCA results with it to form a new set

# In[ ]:

airports2 = airports[['airport', 'year']]


# In[ ]:

airport_pca = airports2.join(Ydf, on=None, how='left')


# In[ ]:

Ydf.head()


# Next, graph the results onto the new feature space

# In[ ]:

graph = airport_pca.plot(kind='scatter', x='PC2', y='PC1', figsize=(16,8))

for i, airport in enumerate(airports['airport']):
    graph.annotate(airport, (airport_pca.iloc[i].PC2, airport_pca.iloc[i].PC1))


# **What does the graph tell us?**

# ## Step 8: Cluster with K-Means

# Set up the k-means clustering analysis. Use the graph from above to derive "k"

# In[ ]:

kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit(x)


# Compute the labels and centroids

# In[ ]:

labels = kmeans.labels_
centroids = kmeans.cluster_centers_


# In[ ]:

print(centroids)


# Compute the Silhoutte Score

# In[ ]:

metrics.silhouette_score(x, labels, metric='euclidean')


# Lastly, plot the new two-dimensional data along with their cluster assignments: 

# In[ ]:

airport_pca['cluster'] = pd.Series(clusters.labels_)


# In[ ]:

graph2 =airport_pca.plot(
    kind='scatter',
    x='PC2',y='PC1',
    c=airport_pca.cluster.astype(np.float), 
    figsize=(16,8))

for i, airport in enumerate(airports['airport']):
    graph.annotate(airport, (airport_pca.iloc[i].PC2, airport_pca.iloc[i].PC1))


# In[ ]:




# In[ ]:



