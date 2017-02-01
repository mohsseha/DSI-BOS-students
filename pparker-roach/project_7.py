
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

# In[85]:

import pandas as pd 
import matplotlib
import numpy as np
import sklearn as sk 
import psycopg2 as psy
import sys
import geojson

from matplotlib import pyplot as plt
from psycopg2 import connect
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn import metrics



get_ipython().magic('matplotlib inline')


# #### Step 2: Examine your data 
# 

# In[60]:

cancellations_raw = pd.read_csv("C:/Users/Pat.NOAGALLERY/Documents/data_sources/airport_cancellations.csv")
operations_raw = pd.read_csv("C:/Users/Pat.NOAGALLERY/Documents/data_sources/Airport_operations.csv")
airports_raw = pd.read_csv("C:/Users/Pat.NOAGALLERY/Documents/data_sources/airports.csv")

cancellations = cancellations_raw.dropna() 
operations = operations_raw.dropna() 
airports = airports_raw.dropna() 
print ("Cancellations\n\n",cancellations.columns)
print ("\nOperations\n\n",operations.columns)
print ("\nAirports\n\n",airports.columns)


# In[61]:

print(airports[airports['Facility Type']!= 'Airport'])


# In[89]:

a = [1,2,3]
b = a.copy()
b.remove(2)
a


# $ \sum_{i=0}^n P(B|A_i) = P(B|A_1)P(A_1) + ... + P(B|A_n)P(A_n)$
# $$P(A|B) = \frac{ A \cap B }{B}$$ 
# 
# uniform
# 
# $\frac{1}{n}$
# 
# bernoulli 
# 
# $\binom{n}{k}\cdot p^{k}(1-p)^{1-k} $
# 
# poisson 
# 
# $\frac{e^{-n}n^{x}}{x!}$
# 
# binomial 
# 
# $\binom{n}{k}\cdot p^kq^{n-k} $
# 

# In[62]:

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

# In[63]:

con = None

con = connect(dbname='postgres', user='postgres', host='localhost', password='root')
con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT) # <-- ADD THIS LINE

dbname = "airport_delays"

cur = con.cursor()

try:
    cur.execute('CREATE DATABASE ' + dbname)
except:
    con = connect(dbname=dbname, user='postgres', host='localhost', password='root')
   

#con.close()


# Load our csv files into tables

# In[64]:

engine = create_engine('postgresql://postgres:root@localhost:5432/'+dbname)

cancellations.to_sql("cancellations", engine, if_exists = "replace")
operations.to_sql("operations", engine, if_exists = "replace")
airports.to_sql("airports", engine, if_exists = "replace")


# Join selected airport.csv columns onto airport_cancellations.csv dataframe.
# 

# In[65]:

joined_cancellations = cancellations
for index, row in airports.iterrows():
    joined_cancellations.ix[joined_cancellations.Airport==row['LocID'], 'AP_NAME'] = row['AP_NAME']
    joined_cancellations.ix[joined_cancellations.Airport==row['LocID'], 'ALIAS'] = row['ALIAS']
    joined_cancellations.ix[joined_cancellations.Airport==row['LocID'], 'FAA REGION'] = row['FAA REGION']
    joined_cancellations.ix[joined_cancellations.Airport==row['LocID'], 'CITY'] = row['CITY']
    joined_cancellations.ix[joined_cancellations.Airport==row['LocID'], 'AP Type'] = row['AP Type']
    joined_cancellations.ix[joined_cancellations.Airport==row['LocID'], 'Latitude'] = row['Latitude']
    joined_cancellations.ix[joined_cancellations.Airport==row['LocID'], 'Longitude'] = row['Longitude']
    joined_cancellations.ix[joined_cancellations.Airport==row['LocID'], 'Boundary Data Available'] = row['Boundary Data Available']
    


# Store the new dataframe as a table

# In[66]:

joined_cancellations.to_sql("joined_cancellations", engine, if_exists = "replace")


# In[ ]:




# Query the database for our intial data

# In[67]:

# cur = con.cursor()
# cur.execute("""SELECT * FROM operations""")
# ap = cur.fetchall()
# ap
ops = pd.read_sql_query("SELECT * FROM operations;", engine)

cur.close()


# #### 1.2 What are the risks and assumptions of our data? 

# 1. We do not know about any correlations between departure delays from one airport on another; i.e., if a flieght arrives late, what impact, if any will it have on the operations of the destination airport.
# 2. There is an assumption that all airports collect and report their data in exactly the same way, year over year.
# 3. 

# ### Part 2: Exploratory Data Analysis

# #### 2.1 Plot and Describe the Data

# In[68]:

ops = ops.drop('index', axis=1)
ops.columns


# #### Are there any unique values? 

# In[69]:

col=ops.columns
len(col)


# ### Part 3: Data Mining

# #### 3.1 Create Dummy Variables

# In[70]:

ops.head()


# In[ ]:




# In[36]:

x = ops.ix[:,2:14].values
y = ops.ix[:,0].values


# In[40]:

xStand = StandardScaler().fit_tran
sform(x)


# In[71]:

covMat1 = np.cov(xStand.T)
eigenValues, eigenVectors = np.linalg.eig(covMat1)
print(covMat1)


# In[43]:

print(eigenValues)


# In[44]:

print(eigenVectors)


# In[45]:

eigenPairs = [(np.abs(eigenValues[i]), eigenVectors[:,i]) for i in range(len(eigenValues))]
eigenPairs.sort()
eigenPairs.reverse()
for i in eigenPairs:
    print(i[0])


# In[47]:

totalEigen = sum(eigenValues)
varExpl = [(i / totalEigen)*100 for i in sorted(eigenValues, reverse=True)]
cumulvarExpl = np.cumsum(varExpl)


# In[48]:

print(cumulvarExpl)


# In[82]:

airports_pca = PCA(n_components=2)
Y = airports_pca.fit_transform(xStand)
Y[0:-1,0]



# In[83]:

# plt.hist(Y)
# plt.title("Distribution of % Airport Departure Delays")
# plt.ylabel("% of Delays")
# plt.xlabel("Airports")
# plt.show()

plt.scatter(Y[0:-1,0], Y[0:-1,1])


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



