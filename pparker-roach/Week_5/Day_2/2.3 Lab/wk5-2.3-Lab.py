
# coding: utf-8

# # Project Pipeline Lab
# 
# 
# ## 1. Data loading
# In this lab we will create pipelines for data processing on the [Titanic dataset](http://www.kaggle.com/c/titanic-gettingStarted/data).
# 
# The dataset is a list of passengers. The second column of the dataset is a “label” for each person indicating whether that person survived (1) or did not survive (0). Here is the Kaggle page with more information on the dataset:
# 
# You can grab the titanic data as follows:
# 
#     psql -h dsi.c20gkj5cvu3l.us-east-1.rds.amazonaws.com -p 5432 -U dsi_student titanic
#     password: gastudents

# In[1]:

import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
engine = create_engine('postgresql://dsi_student:gastudents@dsi.c20gkj5cvu3l.us-east-1.rds.amazonaws.com/titanic')

df = pd.read_sql('SELECT * FROM train', engine)


# Have a look at the data using the info method:
# 
# - Are there numerical features?
# - Are there categorical features?
# - Which columns have missing data?
# - Which of these are important to be filled?

# In[2]:

df.info()


# ## 2. Age
# 
# Several passengers are missing data points for age. Impute the missing values so that there are no “NaN” values for age as inputs to your model. Explore the distribution of age and decide how you are going to impute the data.

# In[3]:

df.Age.plot(kind = 'hist')


# ### 2.b: Age Transformer
# 
# Create a custom transformer that imputes the age values. Depending on how you have decided to impute missing values, this could involve:
# 
# - Selecting one or more columns
# - Filling the NAs using Imputer or a custom strategy
# - Scaling the Age values

# In[4]:

from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def transform(self, X, *_):
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X[self.columns])
        else:
            raise TypeError("This transformer only works with Pandas Dataframes")
    
    def fit(self, X, *_):
        return self
    
cs = ColumnSelector('Age')

cs.transform(df).head()


# In[5]:

age_pipe = make_pipeline(ColumnSelector('Age'),
                         Imputer(),
                         StandardScaler())

age_pipe.fit_transform(df)[:5]


# ## 3. Categorical Variables
# 
# `Embarked` and `Pclass` are categorical variables. Use pandas get_dummies function to create dummy columns corresponding to the values.
# 
# `Embarked` has 2 missing values. Fill them with the most common port of embarkment.
# 
# Feel free to create a GetDummiesTransformer that wraps around the get_dummies function.

# In[6]:

df.Embarked.value_counts()


# In[7]:

df.Embarked = df.Embarked.fillna('S')


# In[8]:

pd.get_dummies(pd.DataFrame(df['Embarked'].head()))


# In[9]:

class GetDummiesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def transform(self, X, *_):
        if isinstance(X, pd.DataFrame):
            return pd.get_dummies(X[self.columns], columns = self.columns)
        else:
            raise TypeError("This transformer only works with Pandas Dataframes")
    
    def fit(self, X, *_):
        return self
    
gdt = GetDummiesTransformer(['Embarked'])
gdt.fit_transform(df.head())


# In[10]:

one_hot_pipe = GetDummiesTransformer(['Pclass', 'Embarked'])

one_hot_pipe.fit_transform(df).head()


# ## 4. Boolean Columns
# 
# The `Sex` column only contains 2 values: `male` and `female`. Build a custom transformers that is initialized with one of the values and returns a boolean column with values of `True` when that value is found and `False` otherwise.

# In[11]:

class TrueFalseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, flag):
        self.flag = flag
    
    def transform(self, X, *_):
        return X == self.flag

    def fit(self, X, *_):
        return self


# In[12]:

gender_pipe = make_pipeline(ColumnSelector('Sex'),
                            TrueFalseTransformer('male'))

gender_pipe.fit_transform(df.head())


# ## 5. Fare
# 
# The `Fare` attribute can be scaled using one of the scalers from the preprocessing module. 

# In[13]:

fare_pipe = make_pipeline(ColumnSelector('Fare'),
                          StandardScaler())

fare_pipe.fit_transform(df.head())


# ## 6. Union
# 
# Use the `make_union` function from the `sklearn.pipeline` modeule to combine all the pipes you have created.

# In[14]:

union = make_union(age_pipe,
                   one_hot_pipe,
                   gender_pipe,
                   fare_pipe)

union.fit_transform(df.head())


# In[15]:

X = df[[u'Pclass', u'Sex', u'Age', u'SibSp', u'Parch', u'Fare', u'Embarked']]
y = df[u'Survived']


# In[16]:

X = union.fit_transform(X)


# The union you have created is a complete pre-processing pipeline that takes the original titanic dataset and extracts a bunch of features out of it. The last step of this process is to persist the `union` object to disk, so that it can be used again later. The following lines achieve this:

# In[17]:

import dill
import gzip

with gzip.open('C:/Users/Pat.NOAGALLERY/Documents/data_sources/union.dill.gz', 'w') as fout:
    dill.dump(union, fout)


# ## Bonus
# 
# Can you think of a way to engineer an additional boolean feature that keeps track whethere the person is travelling alone or with family?

# In[18]:

df['Family'] = (df['Parch'] > 0 ) & (df['SibSp'] > 0)


# In[ ]:



