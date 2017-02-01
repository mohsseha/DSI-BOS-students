
# coding: utf-8

# Today we're going to utilize a very simple (but rich) data set housed in the UCI Machine Learning repository. The Adult Income Dataset is taken from US Census information and is formatted particularly well to study the features/regressors/predictors that go into determining whether an adult US resident is 'likely' to have a household income greater than $50,000. 
# 
# The data includes age, workclass, a weight variable (to account for the unbalanced sampling), education level, time spent in education (in years), marital status, occupation, relationship, race, sex, individuals residency, and a target column that indicates whether the person attained a household income greater than $50,000. All in all, an interested data set for socio-economic research. So let's get our hands dirty and load up some data!

# In[109]:

from sklearn import naive_bayes
import pandas as pd
import numpy as np
import matplotlib as plt


# # Load the data 

# Load the adult data set, which is just .txt file. There are no column labels. Read the docs for the data set here: https://archive.ics.uci.edu/ml/datasets/Adult, and use the in-built Pandas dataframe options to attach the column labels into the data frame. 

# In[ ]:




# # Convert the categorical variables into unordered integral values

# For us to use the scikit-learn (although not every implementation of) Naive Bayes, we must pass in numerical data. Since we have decided to analyze all unordered categorical values, we can do a one-hot encoding to convert our categorical data into a numerical data frame.
# 
# **Note**: Do not use scikit-learn's implementation of One-hot encoding, we want to get you familiar with a bunch of methods, but as you should know by now, there are many ways to do the same thing. If you want, to a challenge, you can write the procedure both from scikit-learn and Pandas method. 

# In[ ]:




# # Challenge Problem: Alternative Encoding Scheme to One-Hot Encoding

# Likewise, beside doing a One-hot encoding, we could also map each string label in our categorical features to a integral value. As we previously leveraged a Pandas data frame method to do the encoding, we are now going to test out a scikit-learn method to impose the integral value encoding. Please check the docs and read up on: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html. Proceed with the encoding and build a Naive Bayes and Logistic classifier for both. Do we get similar results? What should we expect? And why?

# In[ ]:




# # Summarize the data and engage in elementary data exploration

# For some data exploration, use Pandas histogram methods to display the features. 

# In[125]:

# Write histogram functions here, and/or any other data visualizations


# # Partition the data

# Without using any direct method/libraries that would automatically accomplish this, please partition the data set 70/30. You can use anything from the math, pandas, or numpy library, do not use other libraries. 

# In[ ]:




# # Define your feature set and define your target 

# In[ ]:




# # Run Naive Bayes Classifier

# Instantiate the Naive Bayes predictor from scikit-learn with the training data. 

# In[ ]:




# # Check Accuracy / Score for Naive Bayes

# Define the target and feature set for the test data

# In[ ]:




# Score the Naive Bayes classifier on the test data

# In[ ]:




# # Check Accuracy / Score for a Logistic Classifier 

# Define a logistic regression and train it with the feature and target set

# In[ ]:




# Produce the accuracy score of the logistic regression from the test set

# In[ ]:




# Was that what you expected? All we did was remove non categorical variables, and imposed a One-hot encoding, should we have expected the Naive Bayes to underperform the Logistic? Here are some other things you can think about:
# 
# 1. What other metrics outside of simple accuracy can we utilize to measure performance?
# 2. Could some pair-wise correlation between pair-wise features in our feature set have caused an issue with the Naive Bayes? What are the assumptions for Naive Bayes which may cause this to happen? 
# 3. How could we improve the performance of Naive Bayes? 
# 4. What about the numerica features we left out, should we bring them back in? How?
# 
# If you want to expand on your analysis, why not build a correlation matrix, or perhaps print a summary of the logistic regression, would an ANOVA table help in our assessment for this case? 

# In[ ]:



