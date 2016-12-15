
# coding: utf-8

# # Feature Scaling Demo

# In[1]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model, preprocessing


# In[2]:

# Load the Boston Housing dataset
boston = datasets.load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df.head()


# ## Scaling our data
# 
# Let's see what effect scaling our data has on some of the features by picking two features
# that have a large difference in scale.

# In[3]:

xs = df["NOX"]
ys = df["TAX"]

plt.scatter(xs, ys)
plt.xlabel("NOX")
plt.ylabel("TAX")
plt.show()


# ### Standardization
# 
# Let's apply standardization, transforming our data to have mean zero $(\mu = 0)$ and variance 1 $(\sigma^2 = 1)$ by the formula:
# 
# $$ x' = \frac{x - \mu}{\sigma}$$

# In[5]:

xs = df["NOX"]
ys = df["TAX"]
plt.scatter(xs, ys, color='b')
plt.xlabel("NOX")
plt.ylabel("TAX")
plt.show()

xs = df["NOX"]
mean = np.mean(xs)
std = np.std(xs)
xs = [(x - mean) / std for x in xs]

ys = df["TAX"]
mean = np.mean(ys)
std = np.std(ys)
ys = [(y - mean) / std for y in ys]

plt.scatter(xs, ys, color='r')
plt.xlabel("NOX standardized")
plt.ylabel("TAX standardized")
plt.show()


# As you can see, we did not change the shape of the data, just its scale. You can also use scikit-learn to standardize your data.

# In[6]:

from sklearn import preprocessing

xs = preprocessing.scale(df["NOX"])
ys = preprocessing.scale(df["TAX"])

plt.scatter(xs, ys, color='r')
plt.xlabel("NOX standardized")
plt.ylabel("TAX standardized")
plt.show()


# ### Min-Max Scaling
# 
# To Min-Max scale our data, we use the formula:
# 
# $$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

# In[7]:

xs = df["NOX"]
ys = df["TAX"]
plt.scatter(xs, ys, color='b')
plt.xlabel("NOX")
plt.ylabel("TAX")
plt.show()

xs = df["NOX"]
xmin = np.min(xs)
xmax = np.max(xs)
xs = [(x - xmin) / (xmax - xmin) for x in xs]

ys = df["TAX"]
ymin = np.min(ys)
ymax = np.max(ys)
ys = [(y - ymin) / (ymax - ymin) for y in ys]

plt.scatter(xs, ys, color='r')
plt.xlabel("NOX Min-Max Scaled")
plt.ylabel("TAX Min-Max Scaled")
plt.show()


# In[ ]:

We can use scikit-learn to Min-Max Scale.


# In[8]:

from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()

xs = scaler.fit_transform(df[["NOX"]])
ys = scaler.fit_transform(df[["TAX"]])

plt.scatter(xs, ys, color='r')
plt.xlabel("NOX Min-Max Scaled")
plt.ylabel("TAX Min-Max Scaled")
plt.show()


# ### Normalization
# 
# We normalize the data by dividing through by some kind of sum or total. For example, it's common to normalize simply by the (*L1*) sum $|X| = \sum_{x \in X}{x}$ or by the (*L2*) euclidean sum of squares distance  $||X|| = \sqrt{\sum_{x \in X}{x^2}}$:
# 
# $$x' = \frac{x}{|X|}$$

# ## Guided Practice
# 
# Perform normalization by both the L1 and L2 sums and plot as we did for the other scaling methods.
# 
# If you finish early, repeat the exercise [using scikit-learn](http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-normalization).

# In[21]:

xs = df["NOX"]
ys = df["TAX"]
plt.scatter(xs, ys, color='b')
plt.xlabel("NOX")
plt.ylabel("TAX")
plt.show()

xs = df["NOX"]
ys = df["TAX"]
xsum = np.sum(xs)
ysum = np.sum(ys)
# Normalize xs and ys with L1 sum
xs = [(x/xsum) for x in xs]
ys = [(y/ysum) for y in ys]

plt.scatter(xs, ys, color='r')
plt.xlabel("NOX L1 Normalized")
plt.ylabel("TAX L1 Normalized")
plt.show()

xs = df["NOX"]
ys = df["TAX"]
# Normalize xs and ys with L2 sum

plt.scatter(xs, ys, color='g')
plt.xlabel("NOX L2 Normalized")
plt.ylabel("TAX L2 Normalized")
plt.show()

# Sklearn
# Use preprocessing.normalize on xs and ys
xs = df["NOX"]
ys = df["TAX"]
xs = preprocessing.normalize(df["NOX"])
ys = preprocessing.normalize(df["TAX"])
plt.scatter(xs, ys, color='r')
plt.xlabel("NOX L1 Normalized")
plt.ylabel("TAX L1 Normalized")
plt.show()


# ### Independent Practice
# 
# Let's practice linear fits using feature scaling. For each of the three scaling methods we've discussed:
# * Practice scaling and linear fits on the boston housing data using all the data (scaled) versus the target data `boston.target`. Does scaling or normalization affect any of your models? Determine if the model fit score changed. Explain why or why not. (10-20 mins).
# 
# Next:
# * Try some regularized models. Does scaling have a significant effect on the fit? (10 mins)
# * Try some other models from scikit-learn, such as a [SGDRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html).
# It's ok if you are unfamiliar with the model, just follow the example code
# and explore the fit and the effect of scaling. (10 mins)
# * Bonus: try a few extra models like a [support vector machine](http://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html). What do you think
# about the goodness of fit? Scaling is _required_ for this model.
# 
# ### Bonus Exercises
# 
# Using Scikit-learn, fit some other model to the data, for example a regularization model like a Ridge or Lasso, a [SGDRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html), or a [support vector machine](http://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html). Do any of the scaling methods affect the goodness of fit?

# In[30]:

# These are all basically the same, here's one example.
# The linear regression fit score is not affected by scaling since the coefficients adapt.

df = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

prescaler = preprocessing.MinMaxScaler()
X = scaler.fit_transform(df)
y = boston.target

lm = linear_model.LinearRegression()
model = lm.fit(X, y)
predictions = lm.predict(X)

plt.scatter(y, predictions)
plt.xlabel("True Value")
plt.ylabel("Predicted Value")

plt.show()
print ("r^2:", model.score(X, y))



# In[ ]:

# Stochastic Regressor -- scaling makes a huge difference
# linear_model.SGDRegressor()

# Unscaled



# Scaled



# In[ ]:



