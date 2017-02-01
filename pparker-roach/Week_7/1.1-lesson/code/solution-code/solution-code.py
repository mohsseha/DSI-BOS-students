
# coding: utf-8

# ### Section I: Import the data

# In[48]:

get_ipython().magic('matplotlib inline')

import pandas as pd
import seaborn as sb


# In[49]:

mtcars = pd.read_csv("../../assets/datasets/mtcars.csv")
mtcars.head()


# ### Plot the Data

# In[50]:

sb.lmplot(data=mtcars, x='mpg', y='hp')


# In[51]:

sb.factorplot(data=mtcars, x='gear', y='mpg', hue='gear', kind='point')


# ### Extract the features you to use in clustering into a matrix

# In[52]:

X = mtcars[['mpg', 
            'cyl',
            'disp',
            'hp',
            'drat',
            'wt',
            'qsec',
            'vs',
            'am',
            'gear',
            'carb']
          ]


# ### Cluster the data using K-Means Clustering

# Cluster two of the variables of your choice. Choose K based on your plots and the behavior of the data

# In[53]:

from sklearn.cluster import KMeans

cluster_model = KMeans(n_clusters = 5)
cluster_model.fit(X)


# ### Find the Silhoutte Score and plot the features and clusters

# In[54]:

from sklearn.metrics import silhouette_score

silhouette_score(X, cluster_model.labels_, metric='euclidean')


# In[55]:

mtcars['cluster_label'] = cluster_model.labels_

sb.lmplot(
    x = 'mpg',
    y = 'wt',
    hue = 'cluster_label',
    data = mtcars,
    fit_reg=False
)


# In[ ]:




# In[ ]:



