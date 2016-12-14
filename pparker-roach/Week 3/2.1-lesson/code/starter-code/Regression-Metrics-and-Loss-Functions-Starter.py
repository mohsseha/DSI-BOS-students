
# coding: utf-8

# # Regression Metrics and Loss Functions
# 
# We've seen two examples of _loss functions_ earlier in the week in the context of regularization.
# 
# For a model of the form $y = f(x) + \epsilon$ with predictions $\hat{y}_i$ and true values $y_i$, we have:
# 
# * The sum of squared errors:
# $$\text{SSE} = \sum_{i}{\left(\hat{y}_i - y_i \right)^2}$$
# * A Regularlized version:
# If our model parameters are $\theta_i$ and our regularization parameter is $\alpha$, then the loss function took the form:
# $$\text{L} = \sum_{i}{\left(\hat{y}_i - y_i \right)^2 + \alpha \theta_i}$$
# 
# In this lesson we're going to dig deeper into loss functions and their applications. Different loss functions are useful in different scenarios and there are two very popular loss functions that are used in conjuction with regression. In this case they are sometimes referred to as _regression metrics_.
# 
# The first is the _root mean squared error_ or _RMSE_ and it is the mean of the squared errors. If we have $n$ regression points and their predictions, the [RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation) is:
# 
# $$\text{RMSE} = \sqrt{\frac{\sum_{i}{\left(\hat{y}_i - y_i \right)^2}}{n}}$$
# 
# The second is the _mean absolute error_ or _MAE_, and it differs by use of an absolute value instead of a square. The [MAE](https://en.wikipedia.org/wiki/Average_absolute_deviation) is:
# 
# $$\text{MAE} = \frac{\sum_{i}{|\hat{y}_i - y_i |}}{n}$$
# 
# ## Why have different regression metrics?
# 
# You might be thinking, _what's all the fuss about_? It turns out that there are lots of good reasons to use different loss functions. We've seen one -- regularization -- and now we'll consider the effects of outliers on these two metrics.
# 
# First let's try a very simplified statistics problem. Given a dataset, how can we summarize it with a single number? Do you know any ways?
# 
# This is equivalent to fitting a constant model to the data. It turns out that the _mean_ minimizes the RMSE and the _median_ minimizes the MAE. By analogy, when fitting a model, MAE is more tolerant to outliers. In other words, the degree of error of an outlier has a large impact when using RMSE versus the MAE. Since the choice of loss function affects model fit, it's important to consider how you want errors to impact your models.
# 
# **Summary**
# * Use MAE when how far off an error is makes little difference
# * Use RMSE when more extreme errors should have a large impact
# 
# Finally, note that linear regressions with MAE instead of RMSE are called _least absolute deviation_ regressions rather than least squares regressions.
# 
# ### Bonus: Modes
# 
# It turns out the _mode_ minimizes the sum:
# $$\frac{\sum_{i}{|\hat{y}_i - y_i |^{0}}}{n}$$
# where $0^0=0$ and $x^0=1$ otherwise. Can you see why?
# 

# # Guided practice
# 
# Let's compute the RMSE and the MAE for a sample data set. Let's say we had a quadratic function that we fit a line to:

# In[41]:

xs = [-1, 0, 1, 2, 3]
ys = [x*x + 1 for x in xs] # true values
predictions = [2*x for x in xs]
print (ys)
print (predictions)


# First do the calculation by hand to see how large each term is
# .
# 
# .
# 
# .
# 
# .
# 
# .
# 
# .
# 
# .
# 
# .
# 
# .
# 
# .
# 
# .
# 

# In[ ]:




# In[42]:

from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
print ("RMSE:", math.sqrt(mean_squared_error(ys, predictions)))
print ("MAE:", mean_absolute_error(ys, predictions))


# Now let's add an outlier to the data.

# In[43]:

xs.append(4)
ys.append(17)
predictions.append(30)

print ("RMSE:", math.sqrt(mean_squared_error(ys, predictions)))
print ("MAE:", mean_absolute_error(ys, predictions))


# Notice the impact of adding outliers to our data. The effect on the RMSE was large, a factor of 2.23, versus the impact on the MAE with a factor of 1.92. This behavior is expected because RMSE gives more weight to outliers

# # Indepedent Practice
# 
# Let's explore two scenarios to obtain a better understanding of RMSE and MAE. First let's fit two models to the same set of data, the data above. To do the least mean absolute error we will use `statsmodels`.

# In[44]:

get_ipython().magic('matplotlib inline')

from matplotlib import pyplot as plt
# Make the plots bigger
plt.rcParams['figure.figsize'] = 10, 10
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn import linear_model
import statsmodels.formula.api as smf



# In[45]:

# Let's add a few more points
xs.append(2.5)
ys.append(17)

xs.append(1.5)
ys.append(-6)


# In[46]:

df = pd.DataFrame(np.array([xs, ys]).transpose(), columns=["x", "y"])
df.columns = ["x", "y"]
mod = smf.quantreg('y ~ x', df)
res = mod.fit(q=.5)
res.summary()


# This generated a fit of $y = 3 x + 1$. Let's see what a linear regression yields.

# In[47]:

import statsmodels.api as sm

X = np.array(xs).transpose()
X = sm.add_constant(X)
# Fit and summarize OLS model
mod = sm.OLS(ys, X)
res = mod.fit()
res.summary()


# This yielded a fit of $y = 3.4558 x + 0.3844$.
# 
# ### Exercise
# 
# Plot the data with both functions. Which do you think fits the data better?

# In[48]:

f1 = lambda x: 3*x + 1
f2 = lambda x: 3.4558*x + 0.3844

y1 = []
y2 = []


for x in xs:
    y1.append(f1(x))
    y2.append(f2(x))

plt.scatter(xs,ys)
plt.plot(xs,y1,color='k')
plt.plot(xs,y2,color='g')
plt.show()


# Finally, let's explore another scenario. Linear regression has five major assumptions, one of which is called _constant variance_ or _homoscedasticity_. It means that the errors are distributed with the same variance about the best fit line regardless of the value of the independent variables.
# 
# For example, a persistant level of background noise can cause regression metrics to be poorly estimated. Let's take a look.

# In[49]:

import random
from scipy.stats import norm
# Generate some data
xs = list(np.arange(0, 10, 0.1))
ys = [2*x + norm.pdf(0, 1) for x in xs]
# Add random background noise
xs2 = [10 * random.random() for i in range(20)]
ys2 = [20 * random.random() for i in range(20)]

# Plot the data sets
plt.scatter(xs, ys, color='b')
plt.scatter(xs2, ys2, color='r')
plt.show()


# In[50]:

# Combine the data


xs.extend(xs2)


ys.extend(ys2)
df = pd.DataFrame(np.array([xs, ys]).transpose(), columns=['x', 'y'])
# Plot the data sets
plt.scatter(xs, ys, color='b')
plt.scatter(xs2, ys2, color='r')
plt.show()


# In[52]:


# Fit a line to the data
lm = linear_model.LinearRegression()

xs.extend(xs2)
ys.extend(ys2)

df = pd.DataFrame(np.array([xs, ys]).transpose(), columns=['x', 'y'])
X = df[['x']]
y = df['y']

model = lm.fit(X, y)
predictions = model.predict(X)

# Plot the data and the best fit line
## The data
plt.scatter(X, y)
## The line / model
plt.plot(X, predictions)

plt.show()
print ("r^2:", model.score(X, y))
print ("RMSE:", mean_squared_error(ys, predictions))
print ("MAE:", mean_absolute_error(ys, predictions))
print ("Coefficients:", model.coef_, model.intercept_)


# In[53]:

# Now try a MAE regression with statsmodels and plot it.
# You should see a much better fit.

mod = smf.quantreg('y ~ x', df)
res = mod.fit(q=.5)
res.summary()


# In[54]:

plt.scatter(X, y)
## The line / model
plt.plot(X, predictions, color='g')
plt.plot(X, [2*x + 0.2420 for x in xs], color='r')


# In[83]:

import math
math.sqrt(6.8)


# In[ ]:



