
# coding: utf-8

# In[1]:

import sqlite3


# In[2]:

sqlite_db = "test_db.sqlite"
conn = sqlite3.connect(sqlite_db)
c = conn.cursor()


# In[3]:

c.execute("CREATE TABLE houses (field1 integer Primary KEY, sqft INTEGER, bdrms INTGRGER, age INTEGER, price INTEGER)")
conn.commit()


# In[6]:

last_sale = (None, 4000,5,22,619000)
c.execute('INSERT INTO houses values (?,?,?,?,?)', last_sale)
conn.commit()


# In[9]:

recent_sales = {
    (None, 2390 , 4, 34, 319000),
    (None, 1870 , 3, 14, 289000),
    (None, 1505 , 3, 90, 269000)
}
c.executemany('INSERT INTO houses values (?,?,?,?,?)', recent_sales)
conn.commit()


# In[10]:

from numpy import genfromtxt


# In[37]:

data = (genfromtxt('C:/Users/Pat.NOAGALLERY/Documents/data_sources/housing-data.csv', dtype='i8',
                    delimiter=',', skip_header=1)).tolist()

# append a None value to beginning of each sub-list
for d in data:
    d.insert(0, None)


# In[ ]:




# In[38]:

for d in data:
    c.execute('INSERT INTO houses values (?,?,?,?,?)', d)


# In[39]:

conn.commit()


# In[40]:

results = c.execute('Select * from houses where bdrms = 4')
results.fetchall()


# In[41]:

import pandas as pd


# In[42]:

data = pd.read_csv('C:/Users/Pat.NOAGALLERY/Documents/data_sources/housing-data.csv', low_memory=False)


# In[43]:

data.to_sql('houses_pandas', con = conn, if_exists='replace', index=False)


# In[44]:

from pandas.io import sql as pdsql


# In[45]:

pdsql.read_sql('select * from houses_pandas limit 10', con=conn)


# In[47]:

pdsql.read_sql('SELECT sqft, bdrms, price FROM houses_pandas limit 10', con=conn)


# In[49]:

query = '''SELECT
sqft, bdrms, age, price
FROM houses_pandas
WHERE bdrms = 2 and price < 250000
'''
pdsql.read_sql(query, con=conn)


# In[50]:

query = '''SELECT
sqft, bdrms, age
FROM houses_pandas
WHERE age > 60
'''
pdsql.read_sql(query, con=conn)


# In[51]:

query = '''SELECT COUNT(price)
FROM houses_pandas
'''
pdsql.read_sql(query, con=conn)


# In[52]:

query = '''SELECT AVG(sqft), MIN(price), MAX(price)
FROM houses_pandas
WHERE bdrms = 2
'''
pdsql.read_sql(query, con=conn)


# In[54]:

from sqlalchemy import create_engine
import pandas as pd
connect_param = 'postgresql://dsi_student:gastudents@dsi.c20gkj5cvu3l.us-east-1.rds.amazonaws.com:5432/northwind'
engine = create_engine(connect_param)
pd.read_sql("SELECT * FROM pg_catalog.pg_tables WHERE schemaname='public' limit 10", con=engine)


# Questions:
#   
#   1. What's the average price per room for 1 bedroom apartments?
#   
#   *What's the average price per room for 2 bedrooms apartments?
#   
#   *What's the most frequent apartment size (in terms of bedrooms)?
# 
#   *How many are there of that apartment kind?
#   
#   *What fraction of the total number are of that kind?
#   
#   *How old is the oldest 3 bedrooms apartment?
#   
#   *How old is the youngest apartment?
#   
#   *What's the average age for the whole dataset?
#   
#   *What's the average age for each bedroom size?
# 
# Try to answer all these in SQL.

# In[57]:

query = '''SELECT sqft, bdrms, age, price FROM pg_catalog.pg_tables WHERE bdrms = 2 and price < 250000 '''
pd.read_sql(query, con=engine)


# In[ ]:



