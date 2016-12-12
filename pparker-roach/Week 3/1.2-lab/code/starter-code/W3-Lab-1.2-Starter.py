
# coding: utf-8

# # Plotting the Housing Data
# 
# A good first step when working with any new data set is to do some exploratory data analysis, starting with a plots of the data. Let's download the data. There is some information about the [data set](https://archive.ics.uci.edu/ml/datasets/Housing) at the UCI ML repository. It's a good idea to take a look at the dataset description before proceeding.

# In[7]:

get_ipython().magic('matplotlib inline')

# Download the data, save to a file called "housing.data"
import urllib
from urllib.request import urlretrieve
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
urlretrieve (data_url, "housing.data")


# The data file does not contain the column names in the first line, so we'll need to add those in manually. You can find the names and explanations [here](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names). We've extracted the names below for convenience.

# In[8]:

names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
         "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]


# ## Loading the Data
# 
# Now let's use pandas to load the data into a data frame. Note that the data is space separated (rather than the more common comma separated data). Here are the first few lines:
# 
# ```
# 0.00632  18.00   2.310  0  0.5380  6.5750  65.20  4.0900   1  296.0  15.30 396.90   4.98  24.00
# 0.02731   0.00   7.070  0  0.4690  6.4210  78.90  4.9671   2  242.0  17.80 396.90   9.14  21.60
# ```

# In[9]:

import pandas as pd

data = pd.read_csv("housing.data", header=None, names=names, delim_whitespace=True)

# Take a look at the first few rows
data.head()


# ### Knowledge Check:
# 
# How many rows are in the dataset? = 506

# In[11]:

data.shape


# ## Plotting the Data
# 
# We are interested in the house values, given in column "MEDV" as a target for modeling. By plotting each of the other columns against "MEDV" we can get a sense of which variables may be correlated.
# 
# There are many ways we can plot the data, using `pandas`, `matplotlib`, or `seaborn`. In any case, it's nice to import `seaborn` for the improved styling. Let's try using `pandas` first to make a scatter plot of crime (column "CRIM") versus house value ("MEDV").

# In[ ]:

import seaborn as sns

# Plot using pandas
data.plot.scatter(x='CRIM', y='MEDV')


# It looks like there is a relationship. While houses in low crime areas can have a wide range of values, houses in high crime areas appear to have lower values.
# 
# Knowledge checks:
# * How do we select a column of data from a pandas DataFrame?
# * Let's make the same plot with matplotlib.

# In[19]:

# Knowledge Check Solution
from matplotlib import pyplot as plt

# Plot using matplotlib
plt.scatter(data["CRIM"], data["MEDV"])
plt.xlabel("Crime")
plt.ylabel("House Value")
plt.show


# ### Exercises
# 
# Exercise 1: Using `pandas` or `matplotlib` plot the remaining variables against "MEDV" and discuss the relationships you find. Question: Which variables seem to correlate well?
# 
# ### Bonus Exercises
# 
# Exercise 2: Seaborn is very handy for making plots of data for exploratory purposes. Try using `seaborn`'s [pairplots](https://stanford.edu/~mwaskom/software/seaborn/examples/scatterplot_matrix.html) to make similar plots.
# 
# Exercise 3: Improve your plots by including units and better axis labels. You'll need to read the [data set description](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names).
# 
# Exercise 4: Can you find any visual correlations between two variables other than MEDV?

# ### Remaining variables for plotting agains "MEDV" for Exercises above
# "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
# 
#     1. CRIM      per capita crime rate by town
#     2. ZN        proportion of residential land zoned for lots over 
#                  25,000 sq.ft.
#     3. INDUS     proportion of non-retail business acres per town
#     4. CHAS      Charles River dummy variable (= 1 if tract bounds 
#                  river; 0 otherwise)
#     5. NOX       nitric oxides concentration (parts per 10 million)
#     6. RM        average number of rooms per dwelling
#     7. AGE       proportion of owner-occupied units built prior to 1940
#     8. DIS       weighted distances to five Boston employment centres
#     9. RAD       index of accessibility to radial highways
#     10. TAX      full-value property-tax rate per $10,000
#     11. PTRATIO  pupil-teacher ratio by town
#     12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
#                  by town
#     13. LSTAT    % lower status of the population
#     14. MEDV     Median value of owner-occupied homes in $1000's

# In[42]:

# Plot agains Zn  - proportion of residential land zoned for lots over 25,000 sq.ft.
plt.scatter(data["CRIM"], data["ZN"])
plt.xlabel("Crime")
plt.ylabel("Residential Lots Zoned > 25k sq.ft.")
plt.show


# In[43]:

# Plot agains INDUS  - proportion of non-retail business acres per town
plt.scatter(data["CRIM"], data["INDUS"])
plt.xlabel("Crime")
plt.ylabel("proportion of non-retail business acres per town")
plt.show



# In[44]:

# Plot agains CHAS  - Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
plt.scatter(data["CRIM"], data["CHAS"])
plt.xlabel("Crime")
plt.ylabel("Charles River(= 1 if tract bounds river; 0 otherwise)")
plt.show



# In[45]:

# Plot agains  NOX       nitric oxides concentration (parts per 10 million)
plt.scatter(data["CRIM"], data["NOX"])
plt.xlabel("Crime")
plt.ylabel("nitric oxides concentration (parts per 10 million)")
plt.show


# In[46]:

# Plot agains RM        average number of rooms per dwelling
plt.scatter(data["CRIM"], data["RM"])
plt.xlabel("Crime")
plt.ylabel("average number of rooms per dwelling")
plt.show


# In[57]:

# Plot agains AGE       proportion of owner-occupied units built prior to 1940
plt.scatter(data["CRIM"], data["AGE"])
plt.xlabel("Crime")
plt.ylabel("proportion of owner-occupied\n units built prior to 1940")
plt.show


# In[48]:

# Plot agains DIS       weighted distances to five Boston employment centres
plt.scatter(data["CRIM"], data["DIS"])
plt.xlabel("Crime")
plt.ylabel("Weighted distances to five Boston employment centres")
plt.show


# In[50]:

# Plot agains RAD       index of accessibility to radial highways
plt.scatter(data["CRIM"], data["RAD"])
plt.xlabel("Crime")
plt.ylabel("Index of accessibility to radial highways")
plt.show



# In[51]:

# Plot agains TAX      full-value property-tax rate per $10,000
plt.scatter(data["CRIM"], data["TAX"])
plt.xlabel("Crime")
plt.ylabel("Full-value property-tax rate per $10,000")
plt.show



# In[53]:

# Plot agains PTRATIO  pupil-teacher ratio by town
plt.scatter(data["CRIM"], data["PTRATIO"])
plt.xlabel("Crime")
plt.ylabel("pupil-teacher ratio by town")
plt.show



# In[55]:

# Plot agains B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
plt.scatter(data["CRIM"], data["B"])
plt.xlabel("Crime")
plt.ylabel("1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town")
plt.show


# In[56]:

# Plot against LSTAT    % lower status of the population
plt.scatter(data["CRIM"], data["LSTAT"])
plt.xlabel("Crime")
plt.ylabel("% lower status of the population")
plt.show


# In[ ]:



