
# coding: utf-8

# Write a problem statement & describe the goals of your study to be included in the final report
# Pull the data from the IMDB API
# Scrape related IMDB data
# Join API & scraped data in local Postgres
# Use natural language processing to understand the sentiments of users reviewing the movies
# Mine & refine your data
# Construct bagging and boosting ensemble models
# Construct elastic net models
# Perform gridsearch and validation on models
# Present the results of your findings in a formal report to Netflix, including:
# a problem statement,
# summary statistics of the various factors (year, number of ratings, etc.),
# your random forest model,
# and your recommendations for next steps!

# # Executive Summary 

# ## Problem Statement
# The assignment is to examine which factors lead to certain ratings for movies in order to predict that types of movies and individual may like. Netflix has not yet focused on examining the factors that have led to movies having been rated the top movies of all time.
# 
# ## Goal
#    Using machine learning techniques, specifically tree-based ensemble techniques (random forests, bagging, boosting, etc.) identify key factors which contribute to successful movie ratings, and present them using graphs and narratives. 
#    
#    ### Deliverables
#      * Formal Problem Statement (included in this report)
#      * Summary Statistics
#      * The machine learning model use, with supporting code used to generate the findings
#      * Graphics to support the findings
#      * Recommendations for next steps
#     *
#     

# ## Load Libraries 

# In[1]:

import pandas as pd
import numpy as np
from imdbpie import Imdb #if libabry not found, pip install imdbpie from command line 

from IPython.display import Image


# Pull the data from the IMDB API

# In[5]:

imdb = Imdb()
imdb_top = imdb.top_250()
#imdb.search_for_title("The Dark Knight")
imdb_top
data = pd.DataFrame(imdb_top, columns=['can_rate', 'image', 'num_votes', 'rating', 'tconst', 'title', 'type', 'year'])
data.loc[data['type']=='feature']
data['rating'].max()
data['rating'].min()
data['tconst'].head()
data['year'].min()
from sqlalchemy import create_engine
engine = create_engine('postgresql://postgres:root@localhost:5432/test')
data.to_sql('movies', engine)


# Scrape related IMDB data
# ![title](https://i.imgur.com/pDq0n.png)
# https://developers.themoviedb.org/3/people

# 

# In[12]:

top_actors = pd.read_csv("top_100_actors.csv")

top_actors.drop(['created', 'modified'],inplace=True,axis=1)
print(top_actors.columns)
top_actors.iloc[0]['description']


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

# In[167]:

imdb = Imdb({'anonymize': False,
             'locale': 'en_US',
             'exclude_episodes': False})

def movie_tests():
    print((movie.title))
#    print(('keywords', movie.tomatoes))
#    print(('rating votes', movie.rating.ratingvotes))
#    print(('FilmCountry', movie.FilmCountry))
    print(('type', movie.type))
    print(('tagline', movie.tagline))
    print(('rating', movie.rating))
    print(('certification', movie.certification))
    print(('genres', movie.genres))
    print(('runtime', movie.runtime))
    print(('writers summary', movie.writers_summary))
    print(('directors summary', movie.directors_summary))
    print(('creators', movie.creators))
    print(('cast summary', movie.cast_summary))
    print(('full credits', movie.credits))
    print(('cert', movie.certification))

#if __name__ == '__main__':
movie = imdb.get_title_by_id('tt0705926')
#movie_tests()
foo = imdb.search_for_title()


# x = 0
# for i in foo:
#     print(i['title'])
    
# print(x)


# In[140]:


def person_tests():
    print(('name',person.name))
    print(('name',person.name))
#    print(('firstname',person.firstname))
#    print(('gender',person.gender))
    #print(('directed',person.directed))
    #print(('acted',person.acted))
    #print(('filmography', person.filmography))
    #print(('type', person.type))
    #print(('tagline', person.tagline))
    #print(('rating', person.rating))
    #print(('certification', person.certification))
    #print(('genres', person.genres))
    #print(('runtime', person.runtime))
    #print(('writers summary', person.writers_summary))
    #print(('directors summary', person.directors_summary))
    #print(('creators', person.creators))
    #print(('cast summary', person.cast_summary))
    #print(('full credits', person.credits))
    #print(('cert', person.certification))
    
person = imdb.get_person_by_id("nm0000151")
person_tests()


# In[ ]:




# In[ ]:



