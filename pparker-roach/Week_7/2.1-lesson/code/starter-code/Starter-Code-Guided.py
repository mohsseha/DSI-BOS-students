
# coding: utf-8

# ## Lesson 2.1: Guided Practice

# Now that we've gone over the tools for dimensionality reduction, let's put these tools to practice!
# First, let's setup our imports. 

# In[1]:




# And read in our data:

# In[4]:




# Take a Look at the data's head. How is it structured? 

# In[ ]:




# Next, let's break our dataset into x and y values. Remember, our "y" is going to be the class names, which in this case are the varietal names, and "x" is going to be all of the attributes. 

# In[7]:




# Since we don't know what units our data was measured in, let's standardize "x"

# In[8]:




# Next, We'll define the covariance matrix and reduce the dimensions to calculate the eigenvectors and eigenvalues using numpy

# In[9]:




# Once we have these, let's check the eigenvectors and eigenvalues to see what we've returned

# In[ ]:




# In[ ]:




# Leading into principal componant analysis, let's select the highest eigenvalues which will be our principal componants

# In[ ]:




# and sort the values high to low: 

# In[ ]:




# We've found our Eigenvalues (Principal Componants)!

# In[ ]:



