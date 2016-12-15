
# coding: utf-8

# # Data Workflow Lab 2
# 
# Proceed with your analysis of the Project 3 data set. You may need to compute new columns as you proceed. Fit one or more linear models to the data, investigate model fits and outliers, use regularization when appropriate.
# 
# ### Learning objectives
# - Perform exploratory analysis
# - Generate correlation matrix of the features
# - Generate linear regression models
# - Evaluate model fit
# 
# If appropriate for your models and featuers:
# - Use `fit_transform` or [feature selection](http://scikit-learn.org/stable/modules/feature_selection.html) to pick the best features
# - Try different regularization options

# In[5]:

get_ipython().magic('matplotlib inline')

from collections import defaultdict
import datetime

from matplotlib import pyplot as plt
# Make the plots bigger
plt.rcParams['figure.figsize'] = 10, 10
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn import linear_model


# In[6]:

# Load the data from the previous lab
# If you changed the name you'll need to edit the next line
sales = pd.read_csv("sales.csv")
sales.dropna(inplace=True)
del sales["Unnamed: 0"]
# Convert dates
sales["First Date"] = pd.to_datetime(sales["First Date"], format="%Y-%m-%d")
sales["Last Date"] = pd.to_datetime(sales["Last Date"], format="%Y-%m-%d")

sales.head()


# ## Exploratory Analysis
# Make some plots, look at correlations, etc.

# In[7]:

# There are a number of good correlations
sales[[u'2015 Sales', u'2015 Sales mean',u'Price per Liter mean', u'Zip Code',
       u'2015 Volume Sold (Liters)', u'2015 Volume Sold (Liters) mean',
       u'2015 Margin mean', u'2015 Sales Q1', u'2016 Sales Q1']].corr()


# In[8]:

# Perform some exploratory analysis, e.g.
sales.plot.scatter(x="2015 Margin mean", y="2015 Sales Q1")
plt.show()


# In[10]:

# Fit a model

lm = linear_model.LinearRegression()
X = sales[["2015 Sales Q1"]]
y = sales["2015 Sales"]
lm.fit(X, y)
predictions = lm.predict(X)
print ("Model fit:", lm.score(X, y))
print (lm.coef_[0], lm.intercept_)

# Plot the data and the best fit line
plt.scatter(X, y)
plt.plot(X, predictions)
plt.xlabel("Sales 2015 Q1")
plt.ylabel("Sales 2015")
plt.show()


# In[11]:

# Zoom in on the lower corner
plt.scatter(X, y)
plt.plot(X, predictions)
plt.xlabel("2015 Sales Q1")
plt.ylabel("2015 Sales")
plt.xlim(0, 100000)
plt.ylim(0, 400000)
plt.show()


# In[12]:

# Predict Total 2016 sales, compare to 2015

X = sales[["2016 Sales Q1"]]
predictions = lm.predict(X)
total_2016 = sum(predictions)
total_2015 = sum(sales["2015 Sales"])
X2 = sales[["2015 Sales Q1"]]
pred_2015 = sum(lm.predict(X2))

print( "2015 predicted", pred_2015)
print ("2015 actual", total_2015)
print ("2016 predicted", total_2016)


# In[13]:

# Try per zip code to get better resolution
zip_codes = set(sales["Zip Code"].tolist())
models = dict()
all_predictions = []
coefficients = [[],[]]
for zip_code in sorted(zip_codes):
    # Pull out stores in this zip
    zip_sales = sales[sales["Zip Code"] == zip_code]
    lm = linear_model.LinearRegression()
    # Fit a model
    X = zip_sales[["2015 Sales Q1"]]
    y = zip_sales["2015 Sales"]
    lm.fit(X, y)
    models[zip_code] = lm
    predictions2015 = lm.predict(X)
    X2 = zip_sales[["2016 Sales Q1"]]
    predictions2016 = lm.predict(X2)
    all_predictions.append((sum(y), sum(predictions2015), sum(predictions2016)))
    if lm.score(X, y) > 0.8 and lm.score(X, y) < 1:
        coefficients[0].append(lm.coef_[0])
        coefficients[1].append(lm.intercept_)
print ("Averaged model coefficients:", np.mean(coefficients[0]), np.mean(coefficients[1]))


# In[14]:

# Filter out stores that opened or closed throughout the year
# If this wasn't done already
lower_cutoff = pd.Timestamp("20150301")
upper_cutoff = pd.Timestamp("20151001")
mask = (sales['First Date'] < lower_cutoff) & (sales['Last Date'] > upper_cutoff)
sales2 = sales[mask]


# In[21]:

# Fit a model

lm = linear_model.LinearRegression()
X = sales2[["2015 Sales Q1"]]
print (len(X))
y = sales2["2015 Sales"]
lm.fit(X, y)
predictions = lm.predict(X)
print ("Model fit:", lm.score(X, y))

# Plot the data and the best fit line
plt.scatter(X, y)
plt.plot(X, predictions)
plt.xlabel("Sales 2015 Q1")
plt.ylabel("Sales 2015")
plt.xlim(0, 50000)
plt.ylim(0, 200000)
plt.show()


# In[15]:

# Predict Total 2016 sales, compare to 2015

X = sales[["2016 Sales Q1"]]
predictions = lm.predict(X)
total_2016 = sum(predictions)
total_2015 = sum(sales["2015 Sales"])
X2 = sales[["2015 Sales Q1"]]
pred_2015 = sum(lm.predict(X2))

print ("2015 predicted", pred_2015)
print ("2015 actual", total_2015)
print ("2016 predicted", total_2016)


# In[ ]:



