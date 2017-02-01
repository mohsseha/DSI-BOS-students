
# coding: utf-8

# # Logistic Regression Lab
# 
# In the previous lab we have constructed a processing pipeline using `sklearn` for the titanic dataset. At this point you should have a set of features ready for consumption by a Logistic Regression model.
# 
# In this la we will use the pre-processing pipeline you have created and combine it with a classification model.
# 
# 
# We have imported this titanic data into our PostgreSQL instance that you can find connecting here:
# 
#     psql -h dsi.c20gkj5cvu3l.us-east-1.rds.amazonaws.com -p 5432 -U dsi_student titanic
#     password: gastudents

# First of all let's load a few things:
# 
# - standard packages
# - the training set from lab 2.3
# - the union we have saved in lab 2.3

# In[1]:

import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
engine = create_engine('postgresql://dsi_student:gastudents@dsi.c20gkj5cvu3l.us-east-1.rds.amazonaws.com/titanic')

df = pd.read_sql('SELECT * FROM train', engine)


# In[3]:

import gzip
import dill

with gzip.open('C:/Users/Pat.NOAGALLERY/Documents/data_sources/union.dill.gz') as fin:
    union = dill.load(fin)


# Then, let's create the training and test sets:

# In[4]:

X = df[[u'Pclass', u'Sex', u'Age', u'SibSp', u'Parch', u'Fare', u'Embarked']]
y = df['Survived']


# In[5]:

from sklearn.cross_validation import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# ## 1. Model Pipeline
# 
# Combine the union you have created in the previous lab with a LogisticRegression instance. Notice that a `sklearn.pipeline` can have an arbitrary number of transformation steps, but only one, optional, estimator step as the last one in the chain.

# In[6]:

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

model = make_pipeline(union,
                      LogisticRegression())


# ## 2. Train the model
# Use `X_train` and `y_train` to fit the model.
# Use `X_test` to generate predicted values for the target variable and save those in a new variable called `y_pred`.

# In[7]:

model.fit(X_train, y_train)


# In[8]:

y_pred = model.predict(X_test)


# ## 3. Evaluate the model accuracy
# 
# 1. Use the `confusion_matrix` and `classification_report` functions to assess the quality of the model.
# - Embed the results of the `confusion_matrix` in a Pandas dataframe with appropriate column names and index, so that it's easier to understand what kind of error the model is incurring into.
# - Are there more false positives or false negatives? (remember we are trying to predict survival)
# - How does that relate to what the `classification_report` is showing?

# In[9]:

from sklearn.metrics import confusion_matrix, classification_report


# In[10]:

cm = confusion_matrix(y_test, y_pred)
idx = ['Dead', 'Survived']
col = ['Predicted Dead', 'Predicted Survived']
cmdf = pd.DataFrame(cm, index=idx, columns=col)
cmdf


# In[12]:

print (classification_report(y_test, y_pred))


# > Answers:
# 3. There are more False Negatives
# - This is related to the low recall for the `Survived` class

# ## 4. Improving the model
# 
# Can we improve the accuracy of the model?
# 
# One way to do this is to use tune the parameters controlling it.
# 
# You can get a list of all the model parameters using `model.get_params().keys()`.
# 
# Discuss with your team which parameters you could try to change.

# In[ ]:

model.get_params().keys()


# You can systematically probe parameter combinations by using the `GridSearchCV` function. Implement a new classifier that searches the best parameter combination.
# 
# 1. How will you choose the grid granularity?
# 1. How can you prevent the grid to exponentially grow?

# In[ ]:

from sklearn.grid_search import GridSearchCV


# In[ ]:

clf = GridSearchCV(model,
                   param_grid = {"logisticregression__C":[0.01,0.02,0.03,0.05,
                                                          0.1,0.2,0.3,0.5,
                                                          1.0,2.0,3.0,5.0,
                                                          10.0,20.0,30.0,50.0]})


# In[ ]:

clf.fit(X_train, y_train)


# ## 5. Assess the tuned model
# 
# A tuned grid search model stores the best parameter combination and the best estimator as attributes.
# 
# 1. Use these to generate a new prediction vector `y_pred`.
# - Use the `confusion matrix`and `classification_report` to assess the accuracy of the new model.
# - How does the new model compare with the old one?
# - What else could you do to improve the accuracy?

# In[ ]:

clf.best_params_


# In[ ]:

clf.best_score_


# In[ ]:

y_pred = clf.best_estimator_.predict(X_test)


# In[ ]:

cm = confusion_matrix(y_test, y_pred)
idx = ['Dead', 'Survived']
col = ['Predicted Dead', 'Predicted Survived']
cmdf1 = pd.DataFrame(cm, index=idx, columns=col)
print "Simple Logistic Regression:"
print cmdf
print

print "Tuned Logistic Regression:"
print cmdf1
print

print "Tuned VS Simple:"
print cmdf1 - cmdf


# In[ ]:

print classification_report(y_test, y_pred)


# ## Bonus
# 
# What would happen if we used a different scoring function? Would our results change?
# Choose one or two classification metrics from the [sklearn provided metrics](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) and repeat the grid_search. Do your result change?

# In[ ]:



