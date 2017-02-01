
# coding: utf-8

# # Multi-table Datasets - ENRON Archive

# ## 1. Data import
# 
# Connect to the file 'assets/datasets/enron.db' using one of these methods:
# 
# - sqlite3 python package
# - pandas.read_sql
# - SQLite Manager Firefox extension
# 
# Take a look at the database and query the master table. How many Tables are there in the db?
# 
# > Answer:
# There are 3 tables:
# - MessageBase
# - RecipientBase
# - EmployeeBase

# In[2]:

import sqlite3
import pandas as pd
enron_db = 'C:/Users/Pat.NOAGALLERY/Documents/data_sources/enron.db'
conn =sqlite3.connect(enron_db)


# In[3]:

tables = pd.read_sql("SELECT name FROM sqlite_master WHERE TYPE ='table'", con=conn)
tables


# In[4]:

query = '''SELECT * FROM MessageBase LIMIT 10'''
pd.read_sql(query, con=conn)


# In[5]:

query = '''SELECT * FROM RecipientBase LIMIT 10'''
pd.read_sql(query, con=conn)


# In[6]:

query = '''SELECT * FROM EmployeeBase LIMIT 10'''
pd.read_sql(query, con=conn)


# Query the `sqlite_master` table to retrieve the schema of the `EmployeeBase` table.
# 
# 1. What fields are there?
# 1. What's the type of each of them?

# In[7]:


for table in tables['name']:
    print(table)
    for row in conn.execute("pragma table_info("+table+")").fetchall():
        print (row)


# 1. Print the first 5 rows of EmployeeBase table
# 1. Print the first 5 rows of MessageBase table
# 1. Print the first 5 rows of RecipientBase table
# 
# **Hint**  use `SELECT` and `LIMIT`.

# In[8]:

query = '''SELECT * FROM EmployeeBase LIMIT 5'''
print("EmployeeBase")
print(pd.read_sql(query, con=conn))
query = '''SELECT * FROM MessageBase LIMIT 5'''
print("MessageBase")
print(pd.read_sql(query, con=conn))
query = '''SELECT * FROM RecipientBase LIMIT 5'''
print("RecipientBase")
print(pd.read_sql(query, con=conn))


# Import each of the 3 tables to a Pandas Dataframes

# In[9]:

query = '''SELECT * FROM EmployeeBase'''
employee = pd.read_sql(query, con=conn)
print("employee shape ", employee.shape)
query = '''SELECT * FROM MessageBase'''
message = pd.read_sql(query, con=conn)
print("message shape ", message.shape)
query = '''SELECT * FROM RecipientBase'''
recipient = pd.read_sql(query, con=conn)
print("recipient shape ", recipient.shape)


# ## 2. Data Exploration
# 
# Use the 3 dataframes to answer the following questions:
# 
# 1. How many employees are there in the company?
# - How many messages are there in the database?
# - Convert the timestamp column in the messages. When was the oldest message sent? And the newest?
# - Some messages are sent to more than one recipient. Group the messages by message_id and count the number of recepients. Then look at the distribution of recepient numbers.
#     - How many messages have only one recepient?
#     - How many messages have >= 5 recepients?
#     - What's the highest number of recepients?
#     - Who sent the message with the highest number of recepients?
# - Plot the distribution of recepient numbers using Bokeh.

# In[10]:

print ("Total Employees = ", employee.shape[0])
print ("Total Messages = ", message.shape[0])

import datetime
import time
message['time'] = pd.to_datetime(message['unix_time'], unit='s')
print ("Oldest Message = ", min(message.time))
print ("Newest Message = ", max(message.time))

msg = message.groupby('from_eid').size()

print("There were ", msg[msg == 1].count(), " messages with only 1 recipient")
print("There were ", msg[msg >= 5].count(), " messages with >=  5 recipients")
print("The highest number of recipients for a message  ", max(msg))
print("The highest number of recipients for a message was sent by the user with eid ", msg[msg  == max(msg)].index[0])


# The following is the Bokeh distribution

# In[11]:

from bokeh.charts import Histogram
from bokeh.sampledata.autompg import autompg as df
from bokeh.charts import defaults, vplot, hplot, show, output_file

defaults.width = 450
defaults.height = 350

# input options
hist = Histogram(msg, title="Distribution of Recepient Numbers")

hist.xaxis.axis_label = 'Messages'
hist.yaxis.axis_label = '# of Msgs'


output_file("histograms.html")

show(
    vplot(
        hplot(hist)
    )
)


# Rescale to investigate the tail of the curve

# No need to write any code here as the Bokeh graphs are interactive and you can pan in to examine the tail of the curve.

# ## 3. Data Merging
# 
# Use the pandas merge function to combine the information in the 3 dataframes to answer the following questions:
# 
# 1. Are there more Men or Women employees?
# - How is gender distributed across departments?
# - Who is sending more emails? Men or Women?
# - What's the average number of emails sent by each gender?
# - Are there more Juniors or Seniors?
# - Who is sending more emails? Juniors or Seniors?
# - Which department is sending more emails? How does that relate with the number of employees in the department?
# - Who are the top 3 senders of emails? (people who sent out the most emails)
# 
# MessageBase
# 
# 
# (0, 'mid', 'INTEGER', 0, None, 1)
# 
# (1, 'filename', 'TEXT', 0, None, 0)
# 
# (2, 'unix_time', 'INTEGER', 0, None, 0)
# 
# (3, 'subject', 'TEXT', 0, None, 0)
# 
# (4, 'from_eid', 'INTEGER', 0, None, 0)
# 
# RecipientBase
# 
# (0, 'mid', 'INTEGER', 0, None, 1)
# 
# (1, 'rno', 'INTEGER', 0, None, 2)
# 
# (2, 'to_eid', 'INTEGER', 0, None, 0)
# 
# EmployeeBase
# 
# (0, 'eid', 'INTEGER', 0, None, 0)
# 
# 
# (1, 'name', 'TEXT', 0, None, 0)
# 
# (2, 'department', 'TEXT', 0, None, 0)
# 
# (3, 'longdepartment', 'TEXT', 0, None, 0)
# 
# (4, 'title', 'TEXT', 0, None, 0)
# 
# (5, 'gender', 'TEXT', 0, None, 0)
# 
# (6, 'seniority', 'TEXT', 0, None, 0)
# 

# In[142]:

# Are there more Men or Women employees?
gender = employee.groupby('gender').size()
print ("\nThe gender split in the organization is as follows \n",gender)

#How is gender di(stributed across departments?
gender = employee.groupby(['department', 'gender']).size()
print("\nGender is split across departments as follows \n", gender)

#Who is sending more emails? Men or Women?
base = pd.merge(employee, message, left_on='eid', right_on='from_eid')
basegender = base.groupby('gender').size()
print("\nMessages sent split by gender \n", basegender)

# What's the average number of emails sent by each gender?
basegender = base.groupby('gender').mean()
print("\nMessages sent split by gender \n", basegender)


# Answer the following questions regarding received messages:
# 
# - Who is receiving more emails? Men or Women?
# - Who is receiving more emails? Juniors or Seniors?
# - Which department is receiving more emails? How does that relate with the number of employees in the department?
# - Who are the top 5 receivers of emails? (people who received the most emails)

# In[ ]:




# Which employees sent the most 'mass' emails?

# In[ ]:




# Keep exploring the dataset, which other questions would you ask?
# 
# Work in pairs. Give each other a challenge and try to solve it.
