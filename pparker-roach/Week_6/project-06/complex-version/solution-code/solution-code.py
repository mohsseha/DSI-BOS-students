
# coding: utf-8

# ## Project 6

# In[52]:

import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import scipy
import requests
from imdbpie import Imdb
import nltk
import matplotlib.pyplot as plt
import urllib
from bs4 import BeautifulSoup
import nltk
import collections
import re
import csv
import psycopg2

get_ipython().magic('matplotlib inline')


# ### Pre-Work: Write a problem statement 

#     

# ## Part 1: Acquire the Data

# #### 1. Connect to the IMDB API

# In[4]:

imdb = Imdb()
imdb = Imdb(anonymize=True)


# #### 2. Query the top 250 rated movies in the database

# In[5]:

movies = pd.DataFrame(imdb.top_250())


# In[6]:

movies.head()


# #### 3. Only select the top 25 movies and delete the uncessary rows

# In[7]:

df = movies[0:26]


# In[8]:

del df['can_rate']
del df['image']
del df['type']


# In[8]:

df.head()


# #### 4. Write the Results to a csv

# In[47]:

df.to_csv('movies.csv')


# ## Part 2: Wrangle the text data

# #### 1. Convert the listing identification numbers (tconst) from the first dataframe to a list

# In[12]:

numid = df['tconst'].tolist()


# In[13]:

print(numid)


# #### 2. Scrape the reviews for the top 25 movies

# In[29]:

for x in (numid):
    address = ('http://www.imdb.com/title/' + x + '/reviews')
    html = requests.get(address).text
    soup = BeautifulSoup(html, 'html.parser')


# #### 3. Work through each title and find the most common descriptors

# *Hint*: "soup" from BeautifulSoup is the html returned from all 25 pages. You'll need to either address each page individually or break them down by elements

# In[15]:

title = soup.find("title")
page = soup.find_all('p')


# In[18]:

print(title)


# In[17]:

print(page)


# #### 4. Convert to a string and remove the non AlphaNumeric characters

# In[22]:

foo = str(page)


# In[23]:

regex = re.compile('[^a-zA-Z]')
new = regex.sub(' ', foo)


# #### 5. Tokenize the Output

# In[24]:

tokens = nltk.word_tokenize(new)


# In[27]:

counter = collections.Counter(tokens)
movie1 = counter.most_common()
print(movie1)


# #### 6. Convert to a Dataframe for Easy Viewing

# In[30]:

test2 = pd.DataFrame(movie1)


# In[37]:

test2.columns = ['word', 'rank']


# ####  7. Find the rows with the top five descriptive words

# In[40]:

words = ('best', 'hope', 'love', 'beautiful', 'great')


# In[41]:

test3 = test2.loc[test2['word'].isin(words)]


# #### 8. Write the results to a csv

# In[49]:

test3.to_csv('movie1.csv')


# #### 9. Repeat the process for the other top 24 titles

# ## Part 3: Combine Tables in PostgreSQL

# **Instructor Note**: As a backup to the scraping activity, two datasets containing the sentiment analysis, titles + attributes, and a full joined dataset have been included in the repo for this lesson

# #### 1. Import your two .csv data files into your Postgre Database as two different tables

# For ease, we can call these table1 and table2

# #### 2. Connect to database and query the joined set

# In[ ]:

conn = psycopg2.connect(database='yourdb', user='dbuser', password='abcd1234', host='server', port='5432', sslmode='require')
cur = conn.cursor()


# #### 3. Join the two tables 

# In[ ]:

cur.execute("""SELECT *
FROM table1
INNER JOIN table2
ON table1.name = table2.name;""")
     conn.commit()


# #### 4. Select the newly joined table and save two copies of the into dataframes

# In[ ]:

cur.execute("""SELECT * FROM table1""")
df = cur.fetchall()


# In[ ]:

cur.execute("""SELECT * FROM table1""")
df2 = cur.fetchall()
cur.close()


# ## Part 4: Parsing and Exploratory Data Analysis

# #### 1. Rename the column headings

# In[35]:

df.columns = ['votes', 'rating', 'reference', 'title', 'year', 'excellent', 'great', 'love', 'beautiful', 'best', 'hope', 'groundbreaking', 'amazing']


# In[26]:

df2.columns = ['votes', 'rating', 'reference', 'title', 'year', 'excellent', 'great', 'love', 'beautiful', 'best', 'hope', 'groundbreaking', 'amazing']


# In[11]:

df.head()


# #### 2. Run a description of the data

# In[56]:

df.describe()


# #### 3. Visualize the Data

# In[58]:

df.year.hist()
plt.title('Histogram of Years')
plt.xlabel('Year')
plt.ylabel('Frequency')


# In[60]:

plt.plot(df['votes'], df['rating'], 'ro')
plt.title('Age vs Fares')
plt.xlabel('Votes')
plt.ylabel('Rating')


# In[65]:

df.great.hist()
plt.xlabel('Review Described Movie as "Great"; 0=No, 1=Yes')
plt.ylabel('Frequency')


# In[64]:

df.boxplot(column='rating')


# ## Part 3: Build the Decision Tree

# #### 1. What is our target attribute? 

# Our target attribute is the rating - we are looking to predict the rating based on a variety of other attributes; IE: the number of votes, and whether certain common words appeared in the reviews. 

# #### 2. Prepare the data and define the training set

# In[44]:

cols = df[['votes', 'year', 'excellent', 'great', 'love', 'beautiful', 'best', 'hope', 'groundbreaking', 'amazing']]
colsRes = df['rating']


# In[49]:

colsRes = np.array(colsRes).astype('f8')
cols = np.array(cols).astype(int)


# In[50]:

print(colsRes)


# #### 2. Train the Model

# In[53]:

rf = RandomForestRegressor(n_estimators=100) # initialize
rf.fit(cols, colsRes)


# In[54]:

print(rf.feature_importances_)


# #### 43 Set up test data and test the model

# In[55]:

colstest = df2[['votes', 'year', 'excellent', 'great', 'love', 'beautiful', 'best', 'hope', 'groundbreaking', 'amazing']]


# In[56]:

colstest = np.array(cols).astype(int)


# In[61]:

results = rf.predict(colstest)


# #### 5. Check the results

# In[58]:

df2['predictions'] = results


# In[59]:

df2.head()


# #### 6. What is overfitting and how are we at risk? 

# Overfitting is when your model is too complicated for your dataset. This can happen when the model tunes itself to the specific characteristics of your dataset, rather than learning them. In the case of our analysis, overfitting is a risk due to the small nature of the dataset. 
