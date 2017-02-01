
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

# ### <center>Project 7</center> 
# 
# In this project, you will implement the the clustering techniques that you've learned this week. 
# 
# #### http://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236&DB_Short_Name=On-Time
# 
# #### Step 1: Load the python libraries that you will need for this project 

# In[2]:

import pandas as pd 
import matplotlib as plt
import numpy as np
import sklearn as sk 
import psycopg2 as psy

from psycopg2 import connect
import sys
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


get_ipython().magic('matplotlib inline')


# #### Step 2: Examine your data 

# In[3]:

cancellations_raw = pd.read_csv("C:/Users/Pat.NOAGALLERY/Documents/data_sources/airport_cancellations.csv")
operations_raw = pd.read_csv("C:/Users/Pat.NOAGALLERY/Documents/data_sources/Airport_operations.csv")
airports_raw = pd.read_csv("C:/Users/Pat.NOAGALLERY/Documents/data_sources/airports.csv")





cancellations = cancellations_raw.dropna() 
operations = operations_raw.dropna() 
airports = airports_raw.dropna() 
print ("Cancellations\n\n",cancellations.columns)
print ("\nOperations\n\n",operations.columns)
print ("\nAirports\n\n",airports.columns)


# ### Intro: Write a problem statement / aim for this project

# We want to understand the behavior of flight cancellations
# 
# Answer: 

# ### Part 1: Create a PostgreSQL database 

# #### 1. Let's create a database where we can house our airport data

# In[4]:

con = connect(dbname='dummy', user='postgres', host='localhost', password='root')


# postgres -D /usr/local/pgsql/data >logfile 2>&1 
# createdb mydb


# Load our csv files into tables

# In[ ]:




# Join airport_cancellations.csv and airports.csv into one table

# In[ ]:




# Query the database for our intial data

# In[ ]:

cur = conn.cursor()
cur.execute("""SELECT * FROM age""")
ap = cur.fetchall()
print ap


# #### 1.2 What are the risks and assumptions of our data? 

# tbd

# ### Part 2: Exploratory Data Analysis

# #### 2.1 Plot and Describe the Data

# In[ ]:

ap.head()
ap.describe()


# #### Are there any unique values? 

# In[ ]:




# ### Part 3: Data Mining

# #### 3.1 Create Dummy Variables

# In[ ]:




# #### 3.2 Format and Clean the Data

# In[ ]:




# ### Part 4: Define the Data

# #### 4.1 Confirm that the dataset has a normal distribution. How can you tell?

# In[ ]:




# #### 4.2 Find correlations in the data

# In[ ]:




# #### 4.3 What is the value of understanding correlations before PCA? 

# Answer: 

# #### 4.4 Validate your findings using statistical analysis

# In[ ]:




# #### 4.5 How can you improve your overall analysis? 

# Answer: 

# ### Part 5: Perform a PCA and Present Findings

# 5.1 Conduct the PCA

# In[ ]:

# Create a clean data frame 
ap1 = ap[['airport','year','departure cancellations','arrival cancellations']]
print ap1.head()


# In[ ]:




# #### 5.2 Write an analysis plan of your findings 

# Create a writeup on the interpretation of findings including an executive summary with conclusions and next steps

# 

# ### Part 6: Copy your Database to AWS 

# Make sure to properly document all of the features of your dataset

# In[ ]:




# ### Bonus: Create a 3-Dimensional Plot of your new dataset with PCA applied

# In[ ]:




# In[ ]:



