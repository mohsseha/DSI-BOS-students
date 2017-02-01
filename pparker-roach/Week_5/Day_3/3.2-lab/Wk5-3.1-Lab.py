
# coding: utf-8

# # SQL Lab
# 
# In this lab we will learn how to use execute SQL from the ipython notebook and practice some queries on the [Northwind sample database](https://northwinddatabase.codeplex.com/) that we used in Lesson 3.1.
# 
# You can access the data with this command:
# 
#     psql -h dsi.c20gkj5cvu3l.us-east-1.rds.amazonaws.com -p 5432 -U dsi_student northwind
#     password: gastudents
# 

# First of all let's install the ipython-sql extension. You can find instructions [here](https://github.com/catherinedevlin/ipython-sql).

# In[ ]:

# !pip install ipython-sql


# Let's see if it works:

# In[28]:

get_ipython().magic('load_ext sql')


# In[29]:

get_ipython().run_cell_magic('sql', 'postgresql://dsi_student:gastudents@dsi.c20gkj5cvu3l.us-east-1.rds.amazonaws.com/northwind', '        \nselect * from orders limit 5;')


# Nice!!! We can now go ahead with the lab!

# In[30]:

import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


# ## 1: Inspect the database
# 
# If we were connected via console, it would be easy to list all tables using `\dt`. We can however access table information performing a query on the `information_schema.tables` table.
# 
# ### 1.a: List Tables
# 
# 1. write a `SELECT` statement that lists all the tables in the public schema of the `northwind` database, sorted alphabetically

# In[31]:

get_ipython().run_cell_magic('sql', '', "SELECT table_schema,table_name\nFROM information_schema.tables\nWHERE table_schema = 'public'\nORDER BY table_name;")


# ### 1.b: Print Schemas
# 
# The table `INFORMATION_SCHEMA.COLUMNS` contains schema information on each.
# 
# Query it to display schemas of all the public tables. In particular we are interested in the column names and data types. Make sure you only include public schemas to avoid cluttering your results with a bunch of postgres related stuff.

# In[32]:

get_ipython().run_cell_magic('sql', '', "SELECT table_name, column_name, data_type\nFROM information_schema.columns\nWHERE table_schema = 'public'\nORDER by table_name;")


# ### 1.c: Table peek
# 
# Another way of quickly looking at table information is to query the first few rows. Do this for a couple of tables, for example: `orders`, `products`, `usstates`. Display only the first 3 rows.
# 

# In[33]:

get_ipython().run_cell_magic('sql', '', 'select * from orders\nLIMIT 3;\n')


# In[34]:

get_ipython().run_cell_magic('sql', '', 'select * from products\nLIMIT 3;')


# In[35]:

get_ipython().run_cell_magic('sql', '', 'select * from usstates\nLIMIT 3;')


# In[99]:

get_ipython().run_cell_magic('sql', '', 'select * from order_details\nLIMIT 3;')


# As you can see, some tables (like `usstates` or `region`) contain information that is probably less prone to change than other tables (like `orders` or `order_details`). This database is well organized to avoid unnecessary duplication. Let's start digging deeper in the data.

# ## 2: Products
# 
# What products is this company selling? The `products` and `categories` tables contain information to answer this question.
# 
# Use a combination of SQL queries and Pandas merge to answer the following questions:
# 
# - What categories of products is the company selling?
# - How many products per category does the catalog contain?
# - Let's focus only on products that have not been discontinued => how many products per category?
# - What are the most expensive 5 products (not discontinued)?
# - How many units of each of these 5 products are there in stock?
# - Draw a pie chart of the categories, with slices that have the size of the number of products in that category (use non discontinued products)

# ### 2.a: What categories of products is the company selling?
# 
# Remember that PostgreSQL is case sensitive.

# In[38]:

categories = get_ipython().magic('sql select "CategoryID", "CategoryName", "Description" from categories;')
print(categories)


# ### 2.b: How many products per category does the catalog contain?
# 
# Keep in mind that you can cast a %sql result to a pandas dataframe using the `.DataFrame()` method.

# In[39]:

prod_count = get_ipython().magic("sql SELECT COUNT('CategoryID') FROM products;")
prod_count


# ### 2.c: How many not discontinued products per category?

# In[45]:

discontinued_prods = get_ipython().magic('%sql select count("Discontinued") from products where "Discontinued" = 0;')
print(discontinued_prods)


# ### 2.d: What are the most expensive 5 products (not discontinued)?

# In[47]:

expensive = get_ipython().magic('%sql select "UnitPrice" from products order by "UnitPrice" DESC limit 5;')
expensive


# ### 2.e: How many units of each of these 5 products are there in stock?

# In[48]:

expensive_in_stock = get_ipython().magic('%sql select "UnitPrice", "UnitsInStock" from products order by "UnitPrice" DESC limit 5;')
expensive_in_stock


# ### 2.f: Pie Chart
# 
# Use pandas to make a pie chart plot.

# In[49]:

expensive_in_stock.pie()


# ## 3: Orders
# 
# Now that we have a better understanding of products, let's start digging into orders.
# 
# - How many orders in total?
# - How many orders per year
# - How many orders per quarter
# - Which country is receiving the most orders
# - Which country is receiving the least
# - What's the average shipping time (ShippedDate - OrderDate)
# - What customer is submitting the highest number of orders?
# - What customer is generating the highest revenue (need to pd.merge with order_details)
# - What fraction of the revenue is generated by the top 5 customers?

# ### 3.a: How many orders in total?

# In[50]:

total_orders = get_ipython().magic('%sql SELECT COUNT("OrderID") from orders;')
print(total_orders)


# ### 3.b: How many orders per year?

# In[79]:

import pandas as pd
orders_by_year = get_ipython().magic('%sql SELECT EXTRACT(YEAR FROM "OrderDate") AS OrderYear FROM orders;')
type(orders_by_year)
df = pd.DataFrame.from_records(orders_by_year, columns=["orderyear"])
df.groupby("orderyear").size()


# ### 3.c: How many orders per quarter?
# 
# Make a line plot for these.

# In[84]:

orders_by_qtr = get_ipython().magic('%sql SELECT EXTRACT(QUARTER FROM "OrderDate") AS OrderQtr FROM orders;')

df = pd.DataFrame.from_records(orders_by_qtr, columns=["orderQtr"])

byquarter=df.groupby("orderQtr").size()
byquarter.plot()


# ### 3.d: Which country is receiving the most orders?

# In[91]:

country_orders_most = get_ipython().magic('%sql SELECT Count("OrderID") as count, "ShipCountry" from orders GROUP By "ShipCountry" ORDER by count DESC LIMIT  1;')
country_orders_most


# ### 3.e: Which country is receiving the least?
# 

# In[92]:

country_orders_least = get_ipython().magic('%sql SELECT Count("OrderID") as count, "ShipCountry" from orders GROUP By "ShipCountry" ORDER by count  LIMIT  1;')
country_orders_least


# ### 3.f: What's the average shipping time (ShippedDate - OrderDate)?

# In[96]:

avg_ship_time = get_ipython().magic('%sql SELECT AVG("ShippedDate" - "OrderDate") as avg_ship_in_days from orders;')
avg_ship_time


# ### 3.g: What customer is submitting the highest number of orders?

# In[97]:

customer_orders_most = get_ipython().magic('%sql SELECT Count("OrderID") as count, "CustomerID" from orders GROUP By "CustomerID" ORDER by count DESC LIMIT  1;')
customer_orders_most


# ### 3.h: What customer is generating the highest revenue (need to pd.merge with order_details)?

# In[133]:

orders = get_ipython().magic('%sql select "OrderID", "CustomerID" from orders;')
details = get_ipython().magic('%sql select "OrderID", "UnitPrice", "Quantity", "Discount" from order_details;')
orders_df = orders.DataFrame()
details_df = details.DataFrame()

merged = orders_df.merge(details_df, on=["OrderID"])
merged['total'] = merged['UnitPrice'] * merged['Quantity']
for index, row in merged.iterrows():
    if row['Discount'] != 0.0:
        print(row['Discount'])
        #row['total'] = row['total'] - (row['total'] * row['Discount'])
merged.head(50)


# ### 3.i: What fraction of the revenue is generated by the top 5 customers?
# 
# Compare that with the fraction represented by 5 customers over the total number of customers.

# In[ ]:




# Wow!! 5.5% of the customers generate a third of the revenue!!

# ## Bonus: Other tables
# 
# Investigate the content of other tables. In particular lookt at the `suppliers`, `shippers` and `employees` tables.

# In[ ]:



