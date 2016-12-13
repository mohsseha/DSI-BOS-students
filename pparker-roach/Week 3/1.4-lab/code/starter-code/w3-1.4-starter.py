
# coding: utf-8

# # Linear Regression Practice
# 
# In this notebook we'll practice linear regresssions on a new data set of real estate transactions from Sacramento.
# 
# Start by [downloading](https://trello-attachments.s3.amazonaws.com/5679b2e91535c674fadb2efe/56b39dbfc8bbe91b11d49e9f/bb26a8e51e1bb392f94c7d7f045b875c/Sacramentorealestatetransactions.csv) the data.
# 
# In the next cell load the data using pandas. Once you have a data frame, use `data.head()` to look at the first few rows.

# In[1]:

get_ipython().magic('matplotlib inline')
import pandas as pd

filename = "Sacramentorealestatetransactions.csv"

data = pd.read_csv(filename)

# We need to process the dates to be datetime variables
data["sale_date"] = pd.to_datetime(data["sale_date"])

data.head()


# ## Exploratory Analysis
# 
# Use pandas to look through the data. Plot the variables as histograms or pairs in scatter plots as needed with seaborn until you understand each one.

# In[2]:

import seaborn as sbn


# ## Visualize the Data
# The data set contains a number of variables that may be correlated with the price of the properties. Make plots of the relevant variables versus the column "price". You can use pandas, matplotlib, or seaborn.

# In[ ]:

import seaborn as sns
from matplotlib import pyplot as plt

x = data['variable-name']
y = data['price']

plt.scatter(x, y)
plt.xlabel("Appropriate Axis Label")
plt.ylabel("Property Price")
plt.show()


# ## Regressions
# 
# * Perform a series of regressions on various combinations of the independent variables.
# * Plot the predicted values versus the true values
# * Which combinations gives the best fit?

# In[ ]:

import statsmodels.api as sm


# ## Bonus Exercises
# 
# * Find the best model you can with the three variables
# * Are longitude and latitude useful separately? Try adding each and both to another model and look for mean_squared_error improvement
# * Can you find any significant correlations between the non-price variables? Which ones?

# In[ ]:



