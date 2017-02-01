
# coding: utf-8

# In[ ]:




# ### <center>Project 7</center> 
# 
# In this project, you will implement the the clustering techniques that you've learned this week. 
# 
# #### http://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236&DB_Short_Name=On-Time
# 
# #### Step 1: Load the python libraries that you will need for this project 

# In[11]:

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk 
import sys



get_ipython().magic('matplotlib inline')


# #### Step 2: Examine your data 

# In[12]:

cancellations_raw = pd.read_csv("C:/Users/Pat.NOAGALLERY/Documents/data_sources/airport_cancellations.csv")
operations_raw = pd.read_csv("C:/Users/Pat.NOAGALLERY/Documents/data_sources/Airport_operations.csv")
airports_raw = pd.read_csv("C:/Users/Pat.NOAGALLERY/Documents/data_sources/airports.csv")





cancellations = cancellations_raw.dropna() 
operations = operations_raw.dropna() 
airports = airports_raw.dropna() 
print ("Cancellations\n\n",cancellations.columns)
print(cancellations.head(2))
print ("\nOperations\n\n",operations.columns)
print(operations.head(2))
print ("\nAirports\n\n",airports.columns)
print(airports.head(2))


# In[17]:

print(airports[airports['Facility Type']!= 'Airport'])


# In[13]:

#(1-operations[['percent on-time gate departures']]).hist()
plt.hist((1-operations['percent on-time airport departures']))
plt.title("Distribution of % Airport Departure Delays")
plt.ylabel("% of Delays")
plt.xlabel("Airports")
plt.show()


# ### Intro: Write a problem statement / aim for this project

# We want to understand the behavior of flight cancellations
# 
# Answer: 

# ### Part 1: Create a PostgreSQL database 

# #### 1. Let's create a database where we can house our airport data

# In[7]:

import psycopg2 as psy
from psycopg2 import connect
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
con = None

con = connect(dbname='postgres', user='postgres', host='localhost', password='root')
con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT) # <-- ADD THIS LINE

dbname = "airport_delays"

cur = con.cursor()

try:
    cur.execute('CREATE DATABASE ' + dbname)
except:
    con = connect(dbname=dbname, user='postgres', host='localhost', password='root')
   




#     cur.execute("select * from pg_database where datname = %(dname)s", {'dname': dbname })
#     answer = cur.fetchall()
#     if len(answer) > 0:
#         print "Database {} exists".format(dbname)
#     else:
#         print "Database {} does NOT exist".format(dbname)
# except Exception, e:
#     print "Error %s" %e
#     sys.exit(1)
# finally:
#     if con:
#         con.close()
# con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
# cur = con.cursor()
# cur.execute('CREATE DATABASE ' + dbname)
cur.close()
#con.close()
# # postgres -D /usr/local/pgsql/data >logfile 2>&1 
# createdb mydb


# In[9]:

from sqlalchemy import create_engine
engine = create_engine('postgresql://postgres:root@localhost:5432/'+dbname)


# Load our csv files into tables

# In[10]:

engine = create_engine('postgresql://postgres:root@localhost:5432/'+dbname)

cancellations.to_sql("cancellations", engine, if_exists = "replace")
operations.to_sql("operations", engine, if_exists = "replace")
airports.to_sql("airports", engine, if_exists = "replace")


# Join airport_cancellations.csv and airports.csv into one table

# In[ ]:




# Query the database for our intial data

# In[38]:

cur = conn.cursor()
cur.execute("""SELECT * FROM age""")
ap = cur.fetchall()
print ap


# #### 1.2 What are the risks and assumptions of our data? 

# 

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

# #### 5.1 Conduct the PCA

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
