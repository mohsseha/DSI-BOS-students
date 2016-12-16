
# coding: utf-8

# # Regularization Mathematically
# 
# Recall that when we form a linear model, we select the model that minimizes the squared error. For a model of the form
# $$y_i = f(x_i) + e_i$$
# we minimize the sum
# $$\sum_{i}{\left(\hat{y}_i - y_i \right)^2}$$
# This is an example of a _loss function_, a function that measures the cost of inaccurate model predictions. To apply the technique of regularization, we modify the loss function with a term that penalizes more complex models. For example, we could use a loss function of the form:
# $$\sum_{i}{\left(\hat{y}_i - y_i \right)^2 + \alpha \theta_i ^2}$$
# where the vector $\theta$ corresponds to the parameters in our model and $\alpha$ is a parameter that controls penalty strength. Larger $\alpha$ means more of a penalty since it makes the sum larger and we're trying to minimize it.
# 
# The classic example is fitting a polynomial to small amounts of data. Let's see how that works with some sample data.

# ## Scikit-Learn
# 
# If you haven't tried out scikit-learn yet, here's how you fit a linear model. Compare to the code for statsmodels.
# 
# ### Statsmodels
# 
# ```python
# import statsmodels.api as sm
# model = sm.OLS(y, X).fit()
# predictions = model.predict(X)
# 
# # r-squared
# print model.rsquared
# 
# # Print out the statistics
# print model.summary()
# ```
# 
# ### Linear Regression with Scikit-learn
# ```python
# from sklearn import linear_model
# lm = linear_model.LinearRegression()
# 
# model = lm.fit(X, y)
# predictions = lm.predict(X)
# print "r^2:", model.score(X, y)
# ```
# 
# **Imporant Note**
# By default, scikit-learn will include a constant term in its linear regressions. You can disable this by using the `fit_intercept` parameter:
# ```python
# linear_model.LinearRegression(fit_intercept=False) # Default True
# ```
# 
# On the other hand, `statsmodels` does *not* include a constant by default. You can add a constant term to your data like so:
# ```python
# # Prepare some data X
# X = sm.add_constant(X) # Adds a constant column to the X matrix
# sm.OLS(y, X) # fit as usual
# ```

# In[1]:

get_ipython().magic('matplotlib inline')
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
plt.rcParams['figure.figsize'] = 8, 8

# Generate some data
def generate_data():
    xs = np.arange(-5, 5, 1)
    
    data = [(x - random.random(), (x + random.random())**2) for x in xs]
    data.sort()
    xs = [x for (x, y) in data]
    ys = [y for (x, y) in data]
    return xs, ys

xs, ys = generate_data()
plt.scatter(xs, ys)
plt.title("Randomly Generated Data")
plt.show()


# Now we fit a model to the data. If we try to fit a size degree polynomial to the data we should obtain a very overfitted model.

# In[2]:

lm = linear_model.LinearRegression()

# This function from numpy builds a matrix of powers for us
X = np.vander(xs, 3)
y = ys

model = lm.fit(X, y)
predictions = lm.predict(X)

plt.scatter(xs, ys)
plt.title("Randomly Generated Data")
plt.plot(xs, predictions)
plt.show()
print "r^2:", model.score(X, y)


# If we apply our model to a another sample of data we should find that the model is a poor fit.

# In[3]:

xs2, ys2 = generate_data()
X = np.vander(xs2, 3)
predictions = lm.predict(X)

plt.scatter(xs2, ys2)
plt.title("Randomly Generated Dataset #2")
plt.plot(xs2, predictions)
plt.show()
print "r^2:", model.score(X, ys2)


# # Ridge Regularization
# Let's use scikit-learn to run a regression with regularization as we described at the beginning of the notebook. This is called _ridge regression_ and also _Tikhonov regularization_.

# In[4]:

# Note: alpha plays the role of lambda in sklearn (lambda is the notation on e.g. Wikipedia)
rlm = linear_model.Ridge(alpha=4, normalize=True)

# Fit the polynomial again with ridge regularization
X = np.vander(xs, 3)
y = ys
ridge_model = rlm.fit(X, y)
predictions = ridge_model.predict(X)

plt.scatter(xs, ys)
plt.title("Randomly Generated Data")
plt.plot(xs, predictions)
plt.show()
print "r^2:", ridge_model.score(X, ys)


# In[5]:

X = np.vander(xs2, 3)
predictions = ridge_model.predict(X)

plt.scatter(xs2, ys2)
plt.title("Randomly Generated Dataset #2")
plt.plot(xs2, predictions)
plt.show()
print "r^2:", ridge_model.score(X, ys2)


# You should have seen that the ridge fit was not quite as good on the original data but much better on the second set of data. This is because we prevented overfitting by using regularization. If that didn't happen, rerun the notebook to generate new datasets.
# 
# If you'd like to see another example of ridge regularization with linear regression, read through [this example](http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols_ridge_variance.html) on the scikit-learn website.

# # Guided Practice
# 
# You may have noticed that in the previous examples the _hyperparameter_ $\alpha$ was set to be four. This was by design since we suspected overfitting and wanted to a larger regularization effect.
# 
# In general we have to decide how to choose the parameter $\alpha$ and there are "automated" methods. One such method is _cross-validation_ and scikit-learn provides methods to help. For our guided practice, let's explore the ridge model that has built-in [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_%28statistics%29). Typically cross-validation works by splitting up the dataset and training the model on different subsets, testing on the remaining points.
# 
# In this case the model created by the cross-validating ridge regression `RidgeCV` from scikit-learn automatically tries different values of $\alpha$ as well. Run the following code multiple times. You should see that different values of $\alpha$ are chosen by the cross-validator (with mixed results depending on how different the datasets are).

# In[6]:

rlmcv = linear_model.RidgeCV(normalize=True)
xs, ys = generate_data()

# Fit the polynomial again with ridge regularization
X = np.vander(xs, 4)
y = ys
ridge_model = rlmcv.fit(X, y)
predictions = ridge_model.predict(X)

plt.scatter(xs, ys)
plt.title("Randomly Generated Data")
plt.plot(xs, predictions)
plt.show()
print "r^2:", ridge_model.score(X, ys)
print "alpha:", rlmcv.alpha_

X = np.vander(xs2, 4)
predictions = ridge_model.predict(X)

plt.scatter(xs2, ys2)
plt.title("Randomly Generated Dataset #2")
plt.plot(xs2, predictions)
plt.show()
print "r^2:", ridge_model.score(X, ys2)


# # Independent Practice
# 
# Now let's explore the Boston housing data and apply cross-validation. There is an excellent [example](http://scikit-learn.org/stable/auto_examples/plot_cv_predict.html) on the scikit-learn website. Take the code available there and modify it to compare the non-cross-validated fit and the cross-validated fit.

# In[7]:

# Work through the cross-validation example, adding in r^2 calculations.
# Does cross-validation produce a better fit in this case? Why or why not?

import pandas as pd

# Without CV

from sklearn import datasets
from sklearn.cross_validation import cross_val_predict, cross_val_score
from sklearn import linear_model
import matplotlib.pyplot as plt

boston = datasets.load_boston()
X = boston.data
y = boston.target

lr = linear_model.LinearRegression()
lr.fit(boston.data, y)
predicted = lr.predict(boston.data)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
print "r^2:", lr.score(X, y)

# With CV

lr = linear_model.LinearRegression()
predicted = cross_val_predict(lr, X, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
print "r^2:", cross_val_score(lr, X, y, cv=10)

# Once you feel comfortable with it, modify the model to use just the variables RM and LSTAT and repeat

df = pd.DataFrame(boston.data, columns=boston.feature_names)

X = df[["RM", "LSTAT"]]

lr = linear_model.LinearRegression()
predicted = cross_val_predict(lr, X, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
print "r^2:", cross_val_score(lr, X, y, cv=10)


# # Lasso
# 
# Lasso (least absolute shrinkage and selection operator) is another method of applying regularization. By this point you should be able to modify the examples above to apply the [Lasso model](http://scikit-learn.org/stable/modules/linear_model.html#lasso) from scikit-learn and the cross-validated version `LassoCV`. The main difference between Lasso and Ridge regularization is how the penalty works. Read through the example and explain how the loss functions are different.
# 
# > The difference is the power used on the parameter term in the loss function:
# 
# $$\sum_{i}{\left(\hat{y}_i - y_i \right)^2 + \alpha |\theta_i|}$$
# 
# instead of
# 
# $$\sum_{i}{\left(\hat{y}_i - y_i \right)^2 + \alpha \theta_i ^2}$$
# 
# **Note**: Since Lasso tries to constrain the size of parameters, it's necessary to scale your data. You can normalize the data by passing `normalize=True` into `Lasso`, or by using the preprocessing methods we covered earlier.
# 
# 
# For the boston dataset the Lasso fit is not quite as good.

# In[8]:

from sklearn import datasets
from sklearn.cross_validation import cross_val_predict, cross_val_score
from sklearn import linear_model
import matplotlib.pyplot as plt

boston = datasets.load_boston()

X = boston.data
y = boston.target

lr = linear_model.Lasso(normalize=True)
lr.fit(boston.data, y)
predicted = lr.predict(boston.data)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
print "r^2:", lr.score(X, y)

# With CV

lr = linear_model.Lasso(normalize=True)
predicted = cross_val_predict(lr, X, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
print "r^2:", cross_val_score(lr, X, y, cv=10)

# Once you feel comfortable with it, modify the model to use just the variables RM and LSTAT and repeat

df = pd.DataFrame(boston.data, columns=boston.feature_names)

X = df[["RM", "LSTAT"]]

lr = linear_model.Lasso(normalize=True)
predicted = cross_val_predict(lr, X, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
print "r^2:", cross_val_score(lr, X, y, cv=10)


# In[ ]:



