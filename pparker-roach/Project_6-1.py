
# coding: utf-8

# # General Assembly Data Science Immersion Program
# # Project 6

# # Executive Summary 

# ## Problem Statement
# The assignment is to examine which factors lead to certain ratings for movies in order to predict that types of movies and individual may like. Netflix has not yet focused on examining the factors that have led to movies having been rated the top movies of all time.
# 
# ## Goal
#    Using machine learning techniques, specifically tree-based ensemble techniques (random forests, bagging, boosting, etc.) identify key factors which contribute to successful movie ratings, and present them using graphs and narratives. 
#    
# ## Deliverables
#     * Formal Problem Statement (included in this report)
#     * Summary Statistics
#     * The machine learning model use, with supporting code used to generate the findings
#     * Graphics to support the findings
#     * Recommendations for next steps   

# ## Summary Statistics

# tbd

# ## The Model

# tbd

# ## Supporting Graphics

# tbd

# ## Recommendations for Next Steps

# * parse the unstructured information in the 'description' field for each actor and pull out various awards, rankings, etc. and see if their inclusion in the models makes a difference
# * find mechanism to include full user reviews of movies to use in feature analysis
# * get the mechanism to store data in a local Postgres database working
# * create a simplified data schema model showing relationship between movies, actors, etc.

# # The Documented Python Model Used For Deriving Results and Recommendations for Next Steps

# #### The following block is used to import all Python libraries used in this model

# In[ ]:

import pandas as pd
import numpy as np
from imdbpie import Imdb #if libabry not found, pip install imdbpie from command line 

from IPython.display import Image


# ### Data Selected
# The Internet Movie Database (abbreviated IMDb) is an online database of information related to films, television programs and video games, including cast, production crew, fictional characters, biographies, plot summaries, trivia and reviews, operated by IMDb.com, Inc., a subsidiary of Amazon.com. There is an IMDb API which direct access to a limited amount of data. Of interest to this work is a feature of the API which returns the Top 250 movies of all time as rated by a proprietary IMDb algorithm. You can also request data using the API on specific movies or actors. This feature will also be used in retrieving data.
# 
# ##### Top 250 Movies of All Time
# Using the API the data returned for a movies in this list is as follows:
#   * can_rate - can this movie be rated? All movies in this list are rated so this will not be used
#   * image - supplied by studio - will also not be used, but here is an example 
# <img src="shawshank.jpg" width="100">
#   * num_votes - the number of votes this movie received - will potentially be used
#   * rating - the ratings for these nighest rated movies of all time range from 8.0 to 9.3 - will be used as our target
#   * tconst - the unique identifier for a movie - this key will be used in a variety of ways in constructing this model
#   * title - the "string" name of the movie - will not be used in the final model
#   * type - all the movies in this list are listed as "feature" so will not be used in this model
#   * year - the year the movie was released starting in 1921 to present - will be converted to a categorigal variable by decade
#     
# 
# ##### Top 100 Actors of all time data fields
# The IMDb website suppliesa list of the Top 100 Actors of All Time in a downloadable CSV format (http://www.imdb.com/list/ls050274118/). The CSV contains the following information:
#   * position - thier placement on the list - 1-100
#   * const - thier unique identifier in the IMDb database - equivalent to the movie's tconst field
#   * created - time/date stamp of when entry - will not be used
#   * modified - time/date stamp of when the entry was last modified - will not be used
#   * description - rich list of data about the actor including
#     - Acting Skill - 1 through 5 stars
#     - Overall Versatilitie - 1 through 5 stars
#     - Role Transformation - 1-5 stars
#     - Oscar Nominations - integer
#     - BAFTA Awards - British Academy of Film and Television Arts - integer
#     - BAFTA Nominations - integer
#     - Golden Globe - integer
#     - Golden Globe Nominations - integer
#     - "movie name" - Level of Difficulty - 1 through 5 stars
#     - Name - first and last name
#     - Known for - url to a movie that they are best known for
#     - Birth date (month/year/day)
#     
# ##### Additional movie data to be retrieved with the API
# Using the "tconst" field returned in the Topp 250 list to retrieve data on individual movies. Data fields that will be retrieved for each movie include:
#   * 
# 
# Each dataset will be initially loaded into Pandas dataframes and then saved as Postgres tables for analysis

# ##### Load the Top 250 Movies of all time into dataframe 'top_250' and drop unwanted columns

# In[ ]:

imdb = Imdb()
imdb_top = imdb.top_250()
#imdb.search_for_title("The Dark Knight")
imdb_top
top_250 = pd.DataFrame(imdb_top, columns=['can_rate', 'image', 'num_votes', 'rating', 'tconst', 'title', 'type', 'year'])
top_250.drop(['can_rate', 'image', 'title', 'type'],inplace=True,axis=1)


# ##### Import the Top 100 Actors and drop unwanted columns

# In[ ]:

top_actors = pd.read_csv("top_100_actors.csv")
top_actors.drop(['created', 'modified'],inplace=True,axis=1)


# ##### Pull selected movie information and add columns to top_250 dataframe

# In[ ]:

for index, row in top_250.iterrows():
    movie = imdb.get_title_by_id(row['tconst'])
    print(movie.genres)
    print(index)
    pd.concat([top_250,pd.DataFrame(columns=['genres'])])
    top_250.ix[index]['genres']=movie.genres
    #top_250[index,'genres'] = movie.genres
    #print(top_250.ix[index]['genres'])
    
#    top_250.set_value(index, 'genres', movie.genres)
#     top_250.iloc[index]['certification'] = movie.certification
#     top_250.iloc[index]['runtime'] = movie.runtime
#     top_250.iloc[index]['writers_summary'] = movie.writers_summary
#     top_250.iloc[index]['directors_summary'] = movie.directors_summary
#     top_250.iloc[index]['creators'] = movie.creators
#     top_250.iloc[index]['cast_summary'] = movie.cast_summary
#     top_250.iloc[index]['credits'] = movie.credits
print(top_250.shape)


# ##### Pull selected actor information and add columns to top_actors dataframe

# In[ ]:




# Join API & scraped data in local Postgres

# In[ ]:




# Use natural language processing to understand the sentiments of users reviewing the movies

# In[ ]:




# Mine & refine your data

# In[ ]:




# Construct bagging and boosting ensemble models

# In[ ]:




# Construct elastic net models

# In[ ]:




# Perform gridsearch and validation on models

# In[ ]:




# Present the results of your findings in a formal report to Netflix, including:
#   * a problem statement,
#   * summary statistics of the various factors (year, number of ratings, etc.),
#   * your random forest model,
#   * and your recommendations for next steps!

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



