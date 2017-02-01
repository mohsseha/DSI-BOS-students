
# coding: utf-8

# # General Assembly Data Science Immersion Program
# # Project X

# # Executive Summary 

# ## Problem Statement
# <>
# 
# 
# ## Goal
# <> 
#    
# ## Deliverables
#     *<>
#     

# # Summary of Findings

# tbd

# ## Supporting Graphics

# tbd

# # Recommendations for Next Steps

# tbd

# ## Data Description
# <>

# ## The Model

# tbd

# * <>
# 

# #### The following block is used to import all Python libraries used in this model

# In[ ]:




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



