
# coding: utf-8

# In[39]:

import pandas as pd
from datetime import timedelta

get_ipython().magic('pylab inline')

df_goog = pd.read_csv('../../assets/datasets/goog.csv')


# Take a high-level look at the data. Describe it. What are we looking at? Hint: We can use our `plot` function to provide a good visual.

# In[40]:

print(df_goog.describe())
print(df_goog.head())
print(df_goog.shape)


# Looking a little deeper, let's gauge the integrity of our data. Is there any cleaning we can do? 

# In[41]:

print(len(df_goog.isnull()))
#looks like all the data is there
df_goog['Date'] = pd.to_datetime(df_goog['Date'])
#let's look at its datatype
for col in list(df_goog):
    print("{} is of type {}".format(col, df_goog[col].dtype))


# Let's examine the Date column. We should probably make it the index for our DataFrame, as we need to order the data by time. Doing this will result in 6 Series objects indexed by DateTime- literal Time Series!

# In[ ]:

df_goog.set_index('Date', inplace=True)


# We need to convert the string to a DateTime object. Pandas has a built in function for this! Easy peasy. We should also ensure that the dates are sorted.

# In[47]:

df_goog.head()


# Let's add some more columns with useful data extracted from the DateTime index.

# In[59]:

df_goog['Year'] = df_goog.index.year

df_goog['Month'] = df_goog.index.month

df_goog['Day'] = df_goog.index.day

df_goog.sort_index(inplace=True)
df_goog


# Let's walk through adding a dummy variable to flag days where the Close price was higher than the Open price

# In[58]:

df_goog["Closed_Up"] = (df_goog['Close']-df_goog['Open']) > 0
df_goog.head()


# We can use the DateTime object to access various different cuts of data using date attributes. For example, if we wanted to get all of the cuts from 2015, we would do as such:

# In[ ]:




# Let's recall the TimeDelta object. We can use this to shift our entire index by a given offset.

# In[ ]:




# On your own, try to shift the entire time series **both** forwards and backwards by the following intervals:
# - 1 hour
# - 3 days
# - 12 years, 1 hour, and 43 seconds

# In[ ]:




# ## Discussion: Date ranges and Frequencies

# In[ ]:




# Note that `asfreq` gives us a `method` keyword argument. Backfill, or bfill, will propogate the last valid observation forward. In other words, it will use the value preceding a range of unknown indices to fill in the unknowns. Inversely, pad, or ffill, will use the first value succeeding a range of unknown indices to fill in the unknowns.

# Now, let's discuss the following points:
# - What does `asfreq` do?
# - What does `resample` do?
# - What is the difference?
# - When would we want to use each?

# We can also create our own date ranges using a built in function, `date_range`. The `periods` and `freq` keyword arguments grant the user finegrained control over the resulting values. To reset the time data, use the `normalize=True` directive.

# **NOTE:** See Reference B in the lesson notes for all of the available offset aliases

# In[ ]:




# We are also given a Period object, which can be used to represent a time interval. The Period object consists of a start time and an end time, and can be created by providing a start time and a given frequency.

# In[ ]:




# Each of these objects can be used to alter and access data from our DataFrames. We'll try those out in our Independent Practice in a moment.
