
# coding: utf-8

# ## Project 7
# In this project, you will implement the the clustering techniques that you've learned this week.

# #### Step 1: Load the python libraries that you will need for this project

# In[56]:

get_ipython().magic('matplotlib inline')

import pandas as pd 
import matplotlib as plt
import numpy as np
import sklearn as sk 
from scipy.stats import pearsonr, normaltest
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import metrics
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning) 
import os 

os.chdir("/Users/patricksmith/Desktop")


# #### Step 2: Examine your data

# In[12]:

df_raw = pd.read_csv("airport_operations.csv")
df = df_raw.dropna() 
print df.head()


# ## Intro: Write a problem statement / aim for this project

# 

# ## Part 1: Create a PostgreSQL database

# #### 1. Let's create a database where we can house our airport data

# In[ ]:

# Load a new PSQL Bash and start server
psql -U postgres
psql -D /usr/local/pgsql/data # User's CD

# Create a new Database

createdb mydb


# #### 2. Create tables for our data

# In[ ]:

CREATE TABLE airport1 (
    airport varchar(80)
    year date, 
    departure_cancellations int
    arrival_cancellations int
);

CREATE TABLE airport2 (
    airport varchar(80),
    year date,
    average_gate_departure_delay int
    average_taxi_out_time int
);


# #### 3. Load our csv files into tables

# In[ ]:

COPY airport_cancels (airport, year, average_gate_departure_delay, average_taxi_out_time) FROM '.../airport_operations.csv' WITH (FORMAT CSV)
COPY airport_ops (airport, year, departure_cancellations, arrival_cancellations) FROM '.../airport_cancellations.csv' WITH (FORMAT CSV)


# #### 4. Merge the Tables

# In[ ]:

SELECT * from airport_ops
LEFT JOIN airport_cancels
ON
  airport_cancels.date = airport_ops.date AND airport_cancels.airport = airport_ops.airport


# #### 5. Query the database for our intial data

# In[ ]:

cur = conn.cursor()
cur.execute("""SELECT * FROM airport_ops""")
df = cur.fetchall()
print(df)


# #### 6. What are the risks and assumptions of our data?

# Answer: Since we do not know the source of the data, we cannot take the data quality for granted. Any correlation / relationship that we discover upon analyzing the data may be due to a discrepency in human error. Therefore, we assume that our data is accurate.

# ## Part 2: Exploratory Data Analysis

# #### 2.1 Plot and Describe the Data

# In[13]:

df.head(10)


# In[15]:

df.describe()


# In[14]:

df['average_taxi_out_time'].plot(kind='hist', alpha=0.5)


# In[16]:

df.boxplot(column='average_taxi_out_time')


# ## Part 3: Data Mining

# #### Are there any unique values?

# In[20]:

##Check for Unique Values 

df.airport.unique()


# #### 3.1 Create Dummy Variables

# In[22]:

leAir = preprocessing.LabelEncoder()
df.airport = leAir.fit_transform(df.airport)


# In[23]:

df.head()


# #### 3.2 Format the data

# In[30]:

taxi = df['average_taxi_out_time']
gate = df['average_gate_departure_delay']
year = df['year']
taxDel = df['average taxi out delay']
airDel = df['average airborne delay']


# In[40]:

del df['departures for metric computation']
del df['arrivals for metric computation']


# ## Part 4: Refine the Data

# #### 4.1 Confirm that the dataset has a normal distribution. How can you tell?

# In[26]:

normaltest(df)


# In[27]:

df_normalized = preprocessing.normalize(df, norm='l2')


# #### 4.2 Find correlations in the data

# In[35]:

# truncate arrays to same length
taxi1 = taxi[0:806]
gate1 = gate[0:806]
year1 = year[0:806]
taxDel1 = taxDel[0:806]
airDel1 = taxDel[0:806]


# In[36]:

# conduct correlations

pearsonr(taxi1, airDel1)


# In[37]:

pearsonr(year1, airDel1)


# In[38]:

pearsonr(gate1, airDel1)


# In[39]:

pearsonr(taxDel1, airDel1)


# #### 4.3 What is the p-value? What does this mean?

# The p-value is our measure of the strength of the relationship between x and y. It measures the how likely it is that the null hypothesis is true; in the case of a correlation, how likely it is that two arrays between anaylized are not related.
# 

# #### 4.4 What is the value of understanding correlations before PCA?¶

# Answer: Highly correlated variables contribute to a common underlying factor in PCA, the principal componant or eigenvector, which helps us further understand patterns in the data.

# ## Part 5: Perform a PCA and Present Findings

# #### 5.1 Define your "X" and "Y" variables

# In[41]:

x = df.ix[:,1:11].values
y = df.ix[:,0].values


# #### 5.2 Standardize the x values

# In[45]:

xStand = preprocessing.StandardScaler().fit_transform(x)


# #### 5.3 Create the covariance matrix

# In[46]:

covMat = np.cov(xStand.T)
eigenValues, eigenVectors = np.linalg.eig(covMat)


# #### Check the Eigenvalues and Eigenvectors

# In[47]:

print(eigenValues)


# In[48]:

print(eigenVectors)


# #### 5.4 Print the Eigenpairs

# In[50]:

eigenPairs = [(np.abs(eigenValues[i]), eigenVectors[:,i]) for i in range(len(eigenValues))]
eigenPairs.sort()
eigenPairs.reverse()
for i in eigenPairs:
    print(i[0])


# #### 5.5 Calculate the explained variance

# In[51]:

totalEigen = sum(eigenValues)
varExpl = [(i / totalEigen)*100 for i in sorted(eigenValues, reverse=True)]


# In[52]:

print(varExpl)


# In[53]:

cvarex = np.cumsum(varExpl)


# In[54]:

print(cvarex)


# #### 5.6 Conduct the PCA

# In[58]:

PCA_A = PCA(n_components=3)
Y = PCA_A.fit_transform(xStand)


# In[63]:

print Y


# #### 5.7 Create a dataframe from the PCA results

# In[64]:

Ydf = pd.DataFrame(Y, columns=["PC1", "PC2", "PCA3"])


# Now, create a new dataframe that uses the airport and year from the original set and join the PCA results with it to form a new set

# In[66]:

airports2 = df[['airport', 'year']]


# In[67]:

airport_pca = airports2.join(Ydf, on=None, how='left')


# #### 5.8 Graph the results to a new feature space

# In[70]:

graph = airport_pca.plot(kind='scatter', x='PC2', y='PC1', figsize=(16,8))

for i, airport in enumerate(df['airport']):
    graph.annotate(airport, (airport_pca.iloc[i].PC2, airport_pca.iloc[i].PC1))


# #### 5.9 Write an analysis plan of your findings

# Create a writeup on the interpretation of findings including an executive summary with conclusions and next steps

# ## Part 6: Copy your Database to AWS

# Make sure to properly document all of the features of your dataset

# In[ ]:

## Create a PostgreSQL database on AWS using the Amazon database wizard on AWS

## Connect to the DB
psql --host=<DB instance endpoint> --port=<port> --username=<master user name> --password --dbname=<database name> 

## Transfer 
select * into newtable from airport_ops


# ## Bonus:

# #### 1. Conduct the analysis again using different variables from the airport_operations.csv dataset

# #### 2. Create visualization of your data using the ggplot features of matplotlib
