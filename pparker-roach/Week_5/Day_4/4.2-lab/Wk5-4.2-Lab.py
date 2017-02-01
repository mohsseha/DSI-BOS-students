
# coding: utf-8

# # Feature Selection Lab
# 
# In this lab we will explore feature selection on the Titanic Dataset. First of all let's load a few things:
# 
# - Standard packages
# - The training set from lab 2.3
# - The union we have saved in lab 2.3
# 
# 
# You can load the titanic data as follows:
# 
#     psql -h dsi.c20gkj5cvu3l.us-east-1.rds.amazonaws.com -p 5432 -U dsi_student titanic
#     password: gastudents

# In[42]:

import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
engine = create_engine('postgresql://dsi_student:gastudents@dsi.c20gkj5cvu3l.us-east-1.rds.amazonaws.com/titanic')

df = pd.read_sql('SELECT * FROM train', engine)


# In[43]:

import gzip
import dill

with gzip.open('C:/Users/Pat.NOAGALLERY/Documents/data_sources/union.dill.gz') as fin:
    union = dill.load(fin)
    
X = df[[u'Pclass', u'Sex', u'Age', u'SibSp', u'Parch', u'Fare', u'Embarked']]
y = df[u'Survived']

X_transf = union.fit_transform(X)
X_transf


# ## 1 Column names
# 
# Uh oh, we have lost the column names along the way! We need to manually add them:
# - age_pipe => 'scaled_age'
# - one_hot_pipe => 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S'
# - gender_pipe => 'male'
# - fare_pipe => 'scaled_fare'
# 
# Now we need to:
# 
# 1. Create a new pandas dataframe called `Xt` with the appropriate column names and fill it with the `X_transf` data.
# 2. Notice that the current pipeline complitely discards the columns: u'SibSp', u'Parch'. Stack them as they are to the new dataframe
# 

# In[77]:

new_cols = ['scaled_age', 'Pclass_1', 'Pclass_2', 'Pclass_3',
            'Embarked_C', 'Embarked_Q', 'Embarked_S',
            'male', 'scaled_fare']

Xt = pd.DataFrame(X_transf, columns=new_cols)
Xt = pd.concat([Xt, X[[u'SibSp', u'Parch']]], axis = 1)
Xt.head()


# ## 2. Feature selection
# 
# Let's use the `SelectKBest` method in scikit learn to see which are the top 5 features.
# 
# - What are the top 5 features for `Xt`?
# 
# => store them in a variable called `kbest_columns`

# In[85]:

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
selector = SelectKBest(f_classif, k=5)

#print(Xt.shape)
selected_data = selector.fit_transform(Xt, y)
kbest_columns = Xt.columns[selector.get_support()]
#print(selected_data.shape)
#print(kbest_columns)
XtBest = pd.DataFrame(selected_data, columns=kbest_columns)
XtBest.head()


# ## 3. Recursive Feature Elimination
# 
# `Scikit Learn` also offers recursive feature elimination as a class named `RFECV`. Use it in combination with a logistic regression model to see what features would be kept with this method.
# 
# => store them in a variable called `rfecv_columns`

# In[128]:

# user logisticregression as an estimator
estimator = LogisticRegression()
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(Xt, y)
rfecv_columns = Xt.columns[selector.support_]
rfecv_columns
help(RFECV)


# ## 4. Logistic regression coefficients
# 
# Let's see if the Logistic Regression coefficients correspond.
# 
# - Create a logistic regression model
# - Perform grid search over penalty type and C strength in order to find the best parameters
# - Sort the logistic regression coefficients by absolute value. Do the top 5 correspond to those above?
# > Answer: Not completely. That could be due to scaling
# 
# => choose which ones you would keep and store them in a variable called `lr_columns`

# In[142]:

#sklearn.model_selection.GridSearchCV
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV
estimator = LogisticRegression()
parameters = {'C':[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],'penalty':['l1','l2']}
gcv = GridSearchCV(estimator, parameters)
gcv.fit(Xt,y)
coeff3 = pd.DataFrame(gcv.best_estimator_.coef_, columns = Xt.columns)
coeff3
#help(GridSearchCV)


# ## 5. Compare features sets
# 
# Use the `best estimator` from question 4 on the 3 different feature sets:
# 
# - `kbest_columns`
# - `rfecv_columns`
# - `lr_columns`
# - `all_columns`
# 
# Questions:
# 
# - Which scores the highest? (use cross_val_score)
# - Is the difference significant?
# > Answer: Not really
# - discuss in pairs

# In[ ]:




# ## Bonus
# 
# Use a bar chart to display the logistic regression coefficients. Start from the most negative on the left.

# In[ ]:



