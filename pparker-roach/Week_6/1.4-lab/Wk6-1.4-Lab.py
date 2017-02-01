
# coding: utf-8

# # APIs Lab
# In this lab we will practice using APIs to retrieve and store data.

# In[1]:

# Imports at the top
import json
import urllib
import pandas as pd
import numpy as np
import requests
import json
import re
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# ## Exercise 1: Get Data From Sheetsu
# 
# [Sheetsu](https://sheetsu.com/) is an online service that allows you to access any Google spreadsheet from an API. This can be a very powerful way to share a dataset with colleagues as well as to create a mini centralized data storage, that is simpler to edit than a database.
# 
# A Google Spreadsheet with Wine data can be found [here]().
# 
# It can be accessed through sheetsu API at this endpoint: https://sheetsu.com/apis/v1.0/dab55afd
# 
# Questions:
# 
# 1. Use the requests library to access the document. Inspect the response text. What kind of data is it?
# > Answer: it's a JSON string
# - Check the status code of the response object. What code is it?
# > 200
# - Use the appropriate libraries and read functions to read the response into a Pandas Dataframe
# > Possible answers include: pd.read_json and json.loads + pd.Dataframe
# - Once you've imported the data into a dataframe, check the value of the 5th line: what's the price?
# > 6

# In[18]:

# You can either post or get info from this API
api_base_url = 'https://sheetsu.com/apis/v1.0/dab55afd'
api_response = requests.get(api_base_url)
api_response.text[:100]
api_response.status_code

df = pd.read_json(api_base_url)
df.columns
print(df.iloc[4]['Price'])
df.head()
df.shape


# ### Exercise 2: Post Data to Sheetsu
# Now that we've learned how to read data, it'd be great if we could also write data. For this we will need to use a _POST_ request.
# 
# 1. Use the post command to add the following data to the spreadsheet:

# In[17]:

post_data = {
'Grape' : ''
, 'Name' : 'My wonderful wine'
, 'Color' : 'R'
, 'Country' : 'US'
, 'Region' : 'Sonoma'
, 'Vinyard' : ''
, 'Score' : '10'
, 'Consumed In' : '2015'
, 'Vintage' : '1973'
, 'Price' : '200'
}
post_data
requests.post(json=post_data,url=api_base_url)


# 1. What status did you get? How can you check that you actually added the data correctly?
# - In this exercise, your classmates are adding data to the same spreadsheet. What happens because of this? Is it a problem? How could you mitigate it?

# In[ ]:




# ## Exercise 3: Data munging
# 
# Get back to the dataframe you've created in the beginning. Let's do some data munging:
# 
# 1. Search for missing data
#     - Is there any missing data? How do you deal with it?
#     - Is there any data you can just remove?
#     - Are the data types appropriate?
# - Summarize the data 
#     - Try using describe, min, max, mean, var

# In[ ]:




# ## Exercise 4: Feature Extraction
# 
# We would like to use a regression tree to predict the score of a wine. In order to do that, we first need to select and engineer appropriate features.
# 
# - Set the target to be the Score column, drop the rows with no score
# - Use pd.get_dummies to create dummy features for all the text columns
# - Fill the nan values in the numerical columns, using an appropriate method
# - Train a Decision tree regressor on the Score, using a train test split:
#         X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.3, random_state=42)
# - Plot the test values, the predicted values and the residuals
# - Calculate R^2 score
# - Discuss your findings
# 

# In[ ]:




# ## Exercise 5: IMDB Movies
# 
# Sometimes an API doesn't provide all the information we would like to get and we need to be creative.
# Here we will use a combination of scraping and API calls to investigate the ratings and gross earnings of famous movies.

# ## 5.a Get top movies
# 
# The Internet Movie Database contains data about movies. Unfortunately it does not have a public API.
# 
# The page http://www.imdb.com/chart/top contains the list of the top 250 movies of all times. Retrieve the page using the requests library and then parse the html to obtain a list of the `movie_ids` for these movies. You can parse it with regular expression or using a library like `BeautifulSoup`.
# 
# **Hint:** movie_ids look like this: `tt2582802`

# In[ ]:




# ## 5.b Get top movies data
# 
# Although the Internet Movie Database does not have a public API, an open API exists at http://www.omdbapi.com.
# 
# Use this API to retrieve information about each of the 250 movies you have extracted in the previous step.
# - Check the documentation of omdbapi.com to learn how to request movie data by id
# - Define a function that returns a python object with all the information for a given id
# - Iterate on all the IDs and store the results in a list of such objects
# - Create a Pandas Dataframe from the list

# In[ ]:




# ## 5.c Get gross data
# 
# The OMDB API is great, but it does not provide information about Gross Revenue of the movie. We'll revert back to scraping for this.
# 
# - Write a function that retrieves the gross revenue from the entry page at imdb.com
# - The function should handle the exception of when the page doesn't report gross revenue
# - Retrieve the gross revenue for each movie and store it in a separate dataframe

# In[ ]:




# ## 5.d Data munging
# 
# - Now that you have movie information and gross revenue information, let's clean the two datasets.
# - Check if there are null values. Be careful they may appear to be valid strings.
# - Convert the columns to the appropriate formats. In particular handle:
#     - Released
#     - Runtime
#     - year
#     - imdbRating
#     - imdbVotes
# - Merge the data from the two datasets into a single one

# In[ ]:




# ## 5.d Text vectorization
# 
# There are several columns in the data that contain a comma separated list of items, for example the Genre column and the Actors column. Let's transform those to binary columns using the count vectorizer from scikit learn.
# 
# Append these columns to the merged dataframe.
# 
# **Hint:** In order to get the actors name right, you'll have to modify the `token_pattern` in the `CountVectorizer`.

# In[ ]:




# ## Bonus:
# 
# - What are the top 10 grossing movies?
# - Who are the 10 actors that appear in the most movies?
# - What's the average grossing of the movies in which each of these actors appear?
# - What genre is the oldest movie?
# 

# In[ ]:



