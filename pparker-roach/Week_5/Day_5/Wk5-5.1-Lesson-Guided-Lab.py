
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target


# In[2]:

model = SVC(kernel='linear')
model.fit(X, y)


# In[4]:

from sklearn.cross_validation import cross_val_score

cvscores = cross_val_score(model, X, y, cv = 5, n_jobs=-1)
print ("CV score: {:.3} +/- {:.3}".format(cvscores.mean(), cvscores.std()))


# In[5]:

from sklearn.grid_search import GridSearchCV
parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 3, 10]}
clf = GridSearchCV(model, parameters, n_jobs=-1)
clf.fit(X, y)
clf.best_estimator_


# In[ ]:



