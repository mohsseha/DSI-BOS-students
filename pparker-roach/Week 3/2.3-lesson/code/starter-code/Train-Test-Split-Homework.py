
# coding: utf-8

# # Introduction
# 
# We've discussed overfitting in the context of bias and variance, and we've seen some techniques like regularization that are used to avoid overfitting. In this lesson we'll discuss another method for avoid overfitting that is commonly referred to a the _train/test split_. The idea is very similar to cross-validation (indeed it is a type of cross-validation) in that we split the dataset into two subsets:
# * a subset to train our model on, and
# * a subset to test our model's predictions on
# 
# This serves two useful purposes:
# * We prevent overfitting by not using all the data, and
# * We have some remaining data to evaluate our model.
# 
# While it may seem like a relatively simple idea, there are some caveats to putting it into practice. For example, if you are not careful it is easy to take a non-random split. Suppose we have salary data on technical professionals that is composed 80% of data from California and 20% elsewhere and is sorted by state. If we split our data into 80% training data and 20% testing data we ight inadvertantly select all the California data to train and all the non-California data to test. In this case we've still overfit on our data set because we did not sufficiently randomize the data.
# 
# In a situation like this we can use _k-fold cross validation_, which is the same idea applied to more than two subsets. In particular, we partition our data into $k$ subsets and train on $k-1$ one of them. holding the last slice for testing. We can do this for each of the possible $k-1$ subsets.

# # Demo
# Let's explore test-training split with some sample datasets.

# In[10]:

get_ipython().magic('matplotlib inline')

from matplotlib import pyplot as plt
# Make the plots bigger
plt.rcParams['figure.figsize'] = 10, 10

import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split

# Load the Boston Housing dataset
columns = "age sex bmi map tc ldl hdl tch ltg glu".split()
diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data, columns=columns)
y = diabetes.target
# Take a look at the data again

df.head()


# Scikit-learn has a nice function to split a dataset for testing and training called `train_test_split`. The `test_size` keyword argument indicates the proportion of the data that should be held over for testing.

# In[11]:

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.4)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# Now we fit a model on the training data and test on the testing data.

# In[12]:

lm = linear_model.LinearRegression()

model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

## The line / model
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")

print ("Score:", model.score(X_test, y_test))     


# Note that we could always split the data up manually. Here's an example for [this dataset](http://scikit-learn.org/stable/auto_examples/exercises/plot_cv_diabetes.html#example-exercises-plot-cv-diabetes-py) of a manual splitting.
# 
# Now let's try out k-fold cross-validation. Again scikit-learn provides useful functions to do the heavy lifting. The function `cross_val_predict` returns the predicted values for each data point when it's in the testing slice.

# In[14]:

from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics

# Perform 6-fold cross validation
scores = cross_val_score(model, df, y, cv=6)
print ("Cross-validated scores:", scores)
# Make cross validated predictions
predictions = cross_val_predict(model, df, y, cv=6)
plt.scatter(y, predictions)
accuracy = metrics.r2_score(y, predictions)
print ("Cross-Predicted Accuracy:", accuracy)


# In[ ]:




# # Guided Practice
# 
# Use what you've learned to train and test models on the Boston housing data set. If you need a few hints take a look at [this example](http://scikit-learn.org/stable/auto_examples/plot_cv_predict.html) but try your best to make it happen first. Complete the following tasks:
# * Fit a linear model to the Boston Housing data using all the available variables. Perform test training splits of 50:50, 70:30, and 90:10, comparing the scores on test data.
# * For the same setup, perform a $k$-fold cross validation with $k=5$ slices (with cross-validated predictions)

# In[32]:

from sklearn import datasets
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt


boston = datasets.load_boston()
boston_df = pd.DataFrame(boston.data)
boston_target = boston.target

X_train, X_test, y_train, y_test = train_test_split(boston_df, boston_target, test_size=0.5)

lm = linear_model.LinearRegression()

model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

## The line / model
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
score_50_50 = model.score(X_test, y_test)
# For the  setup, perform a k-fold cross validation with k=5 slices (with cross-validated predictions)
scores = cross_val_score(model, boston_df, boston_target, cv=6)
print ("Cross-validated scores:", scores)
# Make cross validated predictions
predictions = cross_val_predict(model, boston_df, boston_target, cv=6)
plt.scatter(y, predictions)
accuracy_50_50 = metrics.r2_score(y, predictions)
plt.show()

 
#####
X_train, X_test, y_train, y_test = train_test_split(boston_df, boston_target, test_size=0.3)

lm = linear_model.LinearRegression()

model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

## The line / model
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
score_70_30 = model.score(X_test, y_test)
# For the  setup, perform a k-fold cross validation with k=5 slices (with cross-validated predictions)
scores = cross_val_score(model, boston_df, boston_target, cv=6)
print ("Cross-validated scores:", scores)
# Make cross validated predictions
predictions = cross_val_predict(model, boston_df, boston_target, cv=6)
plt.scatter(y, predictions)
accuracy_70_30 = metrics.r2_score(y, predictions)
plt.show()


#####
X_train, X_test, y_train, y_test = train_test_split(boston_df, boston_target, test_size=0.1)

lm = linear_model.LinearRegression()

model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

## The line / model
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
score_90_10 = model.score(X_test, y_test)
# For the  setup, perform a k-fold cross validation with k=5 slices (with cross-validated predictions)
scores = cross_val_score(model, boston_df, boston_target, cv=6)
print ("Cross-validated scores:", scores)
# Make cross validated predictions
predictions = cross_val_predict(model, boston_df, boston_target, cv=6)
plt.scatter(y, predictions)
accuracy_90_10 = metrics.r2_score(y, predictions)
plt.show()




print ("Score with 50:50 split:", score_50_50)
print ("Cross-Predicted Accuracy:", accuracy_50_50)
print ("Score with 70:30 split:", score_70_30)
print ("Cross-Predicted Accuracy 70:30:", accuracy_70_30) 
print ("Score with 90:10 split:", score_90_10) 
print ("Cross-Predicted Accuracy:", accuracy_90_10)


# # Independent Practice
# 
# Ultimately we use a test-training split to compare multiple models on the same dataset. This could be comparisons of two linear models, or of completely different models on the same data.
# 
# For your independent practice, fit three different models on the Boston housing data. For example, you could pick three different subsets of variables, one or more polynomial models, or any other model that you like. Then:
# * Fix a testing/training split of the data
# * Train each of your models on the training data
# * Evaluate each of the models on the test data
# * Rank the models by how well they score on the testing data set.
# 
# Bonus tasks:
# * Try a few different splits of the data for the same models. Does your ranking change?
# * Perform a k-fold cross validation and use the cross-validation scores to compare your models. Did this change your rankings?

# ## For the independent practice I am choosing to pick 3 different subsets of the variables. I will repeat the process for each of the subsets
# 
# ### pseudo code
#   * define the varialbe for each of the subsets - here are the variables
#         - CRIM     per capita crime rate by town
#         - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#         - INDUS    proportion of non-retail business acres per town
#         - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#         - NOX      nitric oxides concentration (parts per 10 million)
#         - RM       average number of rooms per dwelling
#         - AGE      proportion of owner-occupied units built prior to 1940
#         - DIS      weighted distances to five Boston employment centres
#         - RAD      index of accessibility to radial highways
#         - TAX      full-value property-tax rate per 10,000 dollar  s       
#         - PTRATIO  pupil-teacher ratio by town
#         - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#         - LSTAT    % lower status of the population
#         
#         The following attribure is stored in boston.target 
#         - MEDV     Median value of owner-occupied homes in thousands of dollars
#     * subset 1 = ["CRIM", "ZN", "INDUS", "CHAS", "NOX"]
#     * subset 2 = ["RM", "AGE", "DIS", "RAD"]
#     * subset 3 = ["TAX", "PTRATIO", "B", "LSTAT"]
#     

# In[56]:

# Setup my 3 subsets
from sklearn import datasets
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt


boston = datasets.load_boston()
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_target = boston.target

#subset 1
boston_df_1 = boston_df[["CRIM", "ZN", "INDUS", "CHAS", "NOX"]]

#subset 2
boston_df_2 = boston_df[["RM", "AGE", "DIS", "RAD"]]

#subset 2
boston_df_3 = boston_df[["TAX", "PTRATIO", "B", "LSTAT"]]


# In[66]:

#Fix a testing/training split of the data
#Train each of your models on the training data
#Evaluate each of the models on the test data
#Rank the models by how well they score on the testing data set.


test_ratios = [0.5, 0.3, 0.1]
for test_size in test_ratios:
    
    #subset 1
    X_train, X_test, y_train, y_test = train_test_split(boston_df_1, boston_target, test_size=test_size)

    lm = linear_model.LinearRegression()

    model = lm.fit(X_train, y_train)
    predictions = lm.predict(X_test)

    ## The line / model
    #plt.scatter(y_test, predictions)
    #plt.xlabel("True Values")
    #plt.ylabel("Predictions")
    #plt.show()
    score_50_50_1 = model.score(X_test, y_test)
    #subset 2
    X_train, X_test, y_train, y_test = train_test_split(boston_df_2, boston_target, test_size=test_size)

    lm = linear_model.LinearRegression()

    model = lm.fit(X_train, y_train)
    predictions = lm.predict(X_test)

    ## The line / model
    #plt.scatter(y_test, predictions)
    #plt.xlabel("True Values")
    #plt.ylabel("Predictions")
    #plt.show()
    score_50_50_2 = model.score(X_test, y_test)

    #subset 1
    X_train, X_test, y_train, y_test = train_test_split(boston_df_3, boston_target, test_size=test_size)

    lm = linear_model.LinearRegression()

    model = lm.fit(X_train, y_train)
    predictions = lm.predict(X_test)

    ## The line / model
    #plt.scatter(y_test, predictions)
    #plt.xlabel("True Values")
    #plt.ylabel("Predictions")
    #plt.show()
    score_50_50_3 = model.score(X_test, y_test)

print("Model evaluation on first subset of variables scores with a test_size of ", test_size, " : ", score_50_50_1)
print("Model evaluation on second subset of variables scores with a test_size of ", test_size, " : ", score_50_50_2)
print("Model evaluation on third subset of variables scores with a test size of ", test_size, " : ", score_50_50_3)


# In[ ]:




# In[ ]:



