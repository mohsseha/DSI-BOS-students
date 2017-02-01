
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

pd.read_sql("SELECT name FROM sqlite_master WHERE TYPE ='table'", con=conn)


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

for row in conn.execute("pragma table_info('EmployeeBase')").fetchall():
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

query = '''SELECT * FROM MessageBase'''
message = pd.read_sql(query, con=conn)

query = '''SELECT * FROM RecipientBase'''
recipient = pd.read_sql(query, con=conn)


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

import numpy as np
import scipy.special

from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
p1 = figure(title="Normal Distribution (μ=0, σ=0.5)",tools="save",
            background_fill_color="#E8DDCB")

mu, sigma = 0, 0.5

measured = np.random.normal(mu, sigma, 1000)
hist, edges = np.histogram(measured, density=True, bins=50)

x = np.linspace(-2, 2, 1000)
pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))
cdf = (1+scipy.special.erf((x-mu)/np.sqrt(2*sigma**2)))/2

p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
        fill_color="#036564", line_color="#033649")
p1.line(x, pdf, line_color="#D95B43", line_width=8, alpha=0.7, legend="PDF")
p1.line(x, cdf, line_color="white", line_width=2, alpha=0.7, legend="CDF")

p1.legend.location = "top_left"
p1.xaxis.axis_label = 'x'
p1.yaxis.axis_label = 'Pr(x)'

show(gridplot(p1, ncols=2, plot_width=400, plot_height=400, toolbar_location=None))


# Rescale to investigate the tail of the curve

# In[ ]:




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

# In[ ]:




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
