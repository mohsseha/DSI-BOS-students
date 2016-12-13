
# coding: utf-8

# # Linear Regression with Statsmodels and Scikit-Learn

# Let's investigate the housing dataset with linear regression. Here's the documentation for `statsmodels` (in case you need it):
# * statsmodels -- [linear regression](http://statsmodels.sourceforge.net/devel/examples/#regression)

# ## Intro to Statsmodels
# 
# Statsmodels is a python package that provides access to many useful statistical calculations and models such as linear regression. It has some advantages over `scikit-learn`, in particular easier access to various statistical aspects of linear regression.
# 
# First let's load and explore our dataset, then we'll see how to use statsmodels. We'll use `sklearn` to provide the data.

# In[31]:

get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt

from sklearn import datasets
data = datasets.load_boston()

print (data.DESCR)


# Let's take a minute to see what the data looks like.

# In[32]:

print (data.feature_names)
print (data.data[0])
print (data.target[0])


# Scikit-learn has already split off the house value data into the target variable. Let's see how to build a linear regression. First let's put the data into a data frame for convenience, and do a quick check to see that everything loaded correctly.

# In[33]:

import numpy as np
import pandas as pd
df = pd.DataFrame(data.data, columns=data.feature_names)
# Put the target (housing value -- MEDV) in another DataFrame
targets = pd.DataFrame(data.target, columns=["MEDV"])

# Take a look at the first few rows
print (df.head())
print (targets.head())


# Now let's fit a linear model to the data. First let's take a look at some of the variables we identified visually as being linked to house value, RM and LSTAT. Let's look at each individually and then both together.
# 
# Note that statsmodels does not add a constant term by default, so you need to use `X = sm.add_constant(X)` if you want a constant term.

# In[34]:

import statsmodels.api as sm

X = df["RM"]
y = targets["MEDV"]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print out the statistics
model.summary()


# ### Interpreting the Coefficients
# 
# Here the coefficient of 3.634 means that as the `RM` variable increases by 1, the predicted value of `MDEV` increases by 3.634.
# 
# Let's plot the predictions versus the actual values.

# In[54]:

# Plot the model
plt.scatter(predictions, y, s=30, c='r', marker='+', zorder=10)
plt.xlabel("Predicted Values from RM")
plt.ylabel("Actual Values MEDV")
plt.show()
print ("MSE:", model.mse_model)


# **Check**: How does this plot relate to the model? In other words, how are the independent variable (RM) and dependent variable ("MEDV") incorporated?
# 
# Solution: They are used to make the predicted values (the x-axis)
# 
# Let's try it with a constant term now.

# In[36]:

## With a constant

import statsmodels.api as sm

X = df["RM"]
X = sm.add_constant(X)
y = targets["MEDV"]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print out the statistics
model.summary()


# In[37]:

# Plot the model
plt.scatter(predictions, y, s=30, c='r', marker='+', zorder=10)
plt.xlabel("Predicted Values from RM")
plt.ylabel("Actual Values MEDV")
plt.show()
print "MSE:", model.mse_model


# ### Interpreting the Coefficients
# 
# With the constant term the coefficients are different. Without a constant we are forcing our model to go through the origin, but now we have a y-intercept at -34.67. We also changed the slope of the `RM` regressor from 3.634 to 9.1021.
# 
# Next let's try a different predictor, `LSTAT`.
# 

# In[38]:

X = df[["LSTAT"]]
y = targets["MEDV"]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print out the statistics
model.summary()


# In[55]:

# Plot the model
plt.scatter(predictions, y, s=30, c='r', marker='+', zorder=10)
plt.xlabel("Predicted Values from LSTAT")
plt.ylabel("Actual Values MEDV")
plt.show()
print ("MSE:", model.mse_model)


# Finally, let's fit a model using both `RM` and `LSTAT`.

# In[40]:

X = df[["RM", "LSTAT"]]
y = targets["MEDV"]

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

model.summary()


# In[42]:

# Plot the model
plt.scatter(predictions, y, s=30, c='r', marker='+', zorder=10)
plt.xlabel("Predicted Values from RM and LSTAT")
plt.ylabel("Actual Values MEDV")
plt.show()
print ("MSE:", model.mse_model)


# ## Comparing the models
# 
# A perfect fit would yield a straight line when we plot the predicted values versus the true values. We'll quantify the goodness of fit soon.
# 
# ### Exercis
# X = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
# plte
# 
# Run the fit on all the variables with `X = df`. Did this improve the fit versus the previously tested variable combinations? (Use mean squared error).

# In[43]:


X = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
plt.scatter(predictions, y, s=30, c='r', marker='+', zorder=10)
plt.xlabel("Predicted Values from all determinate variable")
plt.ylabel("Actual Values MEDV")
plt.show()


# ## Preparing data with Patsy
# 
# `Patsy` is a python package that makes preparing data a bit easier. It uses a special formula syntax to create the `X` and `y` matrices we use to fit our models with.
# 
# Let's look at a few examples. To get the `X` and `y` matrices for the previous example, try the following.

# In[44]:

import patsy

# First let's add the targets to our data frame
df["MEDV"] = targets["MEDV"]

y, X = patsy.dmatrices("MEDV ~ RM + LSTAT", data=df)
print (X[0:5, :])
print (y[0:5, :])


# We can also apply functions to our data in the formula. For example, to perform a quadratic regression of "MEDV" with "LSTAT", we would use the following formula.

# In[49]:

y, X = patsy.dmatrices("MEDV ~ LSTAT + I(LSTAT**2)", data=df)
print (X[0:5, :])


# You can use some python functions, like `numpy`'s power.

# In[50]:

y, X = patsy.dmatrices("MEDV ~ LSTAT + np.power(LSTAT,2)", data=df)
print (X[0:5, :])


# Patsy can also handle categorical variables and make dummy variables for you.

# In[51]:

from patsy import dmatrix, demo_data

data = demo_data("a", nlevels=4)
print (data)
dmatrix("a", data)


# ## Guided Practice
# 
# ### Exercises
# 
# Practice using patsy formulas and fit models for
# * CRIM and INDUS versus MDEV (price)
# * AGE and CHAS (categorical) versus MDEV

# In[52]:

y, X = patsy.dmatrices("MEDV ~ CRIM + INDUS", data=df)
print (X[0:5, :])
print (y[0:5, :])


# In[53]:

y, X = patsy.dmatrices("MEDV ~ AGE + CHAS", data=df)
print (X[0:5, :])
print (y[0:5, :])


# ## Independent Practice
# 
# Try to find the best models that you can that:
# * use only two variables
# * only three variables
# * only four variables
# 
# Evaluate your models using the squared error. Which has the lowest? How do the errors compare to using all the variables?

# ### Exercise
# 
# From the LSTAT plot you may have noticed that the relationship is not quite linear. Add a new column `"LSTAT2"` to your data frame for the LSTAT values squared and try to fit a quadratic function using `["LSTAT", "LSTAT2"]`. Is the resulting fit better or worse?

# In[ ]:




# ## Bonus
# 
# We'll go over using Scikit-Learn later this week, but you can get a head start now by repeating some of the exercises using `sklearn` instead of `statsmodels`.
# 
# ### Exercises
# 
# Recreate the model fits above with `scikit-learn`:
# * a model using LSTAT
# * a model using RM and LSTAT
# * a model using all the variables
# 
# Compare the mean squared errors for each model between the two packages. Do they differ significantly? Why or why not?

# In[ ]:



