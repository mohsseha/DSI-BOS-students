
# coding: utf-8

# Today we're going to utilize a very simple (but rich) data set housed in the UCI Machine Learning repository. The Adult Income Dataset is taken from US Census information and is formatted particularly well to study the features/regressors/predictors that go into determining whether an adult US resident is 'likely' to have a household income greater than $50,000. 
# 
# The data includes age, workclass, a weight variable (to account for the unbalanced sampling), education level, time spent in education (in years), marital status, occupation, relationship, race, sex, individuals residency, and a target column that indicates whether the person attained a household income greater than $50,000. All in all, an interested data set for socio-economic research. So let's get our hands dirty and load up some data!

# In[3]:

from sklearn import naive_bayes
import pandas as pd
import numpy as np
import matplotlib as plt


# # Load the data 

# Load the adult data set, which is just .txt file. There are no column labels. Read the docs for the data set here: https://archive.ics.uci.edu/ml/datasets/Adult, and use the in-built Pandas dataframe options to attach the column labels into the data frame. 

# In[5]:

adult_dat = pd.read_csv("adult.txt", names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*', engine='python', na_values="?")

adult_dat.head()


# # Convert the categorical variables into unordered integral values

# For us to use the scikit-learn (although not every implementation of) Naive Bayes, we must pass in numerical data. Since we have decided to analyze all unordered categorical values, we can do a one-hot encoding to convert our categorical data into a numerical data frame.
# 
# **Note**: Do not use scikit-learn's implementation of One-hot encoding, we want to get you familiar with a bunch of methods, but as you should know by now, there are many ways to do the same thing. If you want, to a challenge, you can write the procedure both from scikit-learn and Pandas method. If you want an extra challenge, you can create a function to do it automatically. 

# In[7]:

Sex = pd.get_dummies(adult_dat['Sex'])
Workclass = pd.get_dummies(adult_dat['Workclass']) 
Marital = pd.get_dummies(adult_dat['Martial Status'])
Occupation = pd.get_dummies(adult_dat['Occupation'])
Relationship = pd.get_dummies(adult_dat['Relationship'])
Race = pd.get_dummies(adult_dat['Race'])
Country = pd.get_dummies(adult_dat['Country'])
Target = pd.get_dummies(adult_dat['Target'])

# Clean up the data set by deleting un-used columns

one_hot_dat = pd.concat([adult_dat, Sex, Workclass, Marital, Occupation, Relationship, Race, Country, Target], axis = 1)
del one_hot_dat['Sex']; del one_hot_dat['Age']; del one_hot_dat['Workclass']; del one_hot_dat['fnlwgt']; 
del one_hot_dat['Education']; del one_hot_dat['Education-Num']; del one_hot_dat['Martial Status']
del one_hot_dat['Occupation']; del one_hot_dat['Relationship']; del one_hot_dat['Race']; del one_hot_dat['Capital Gain']
del one_hot_dat['Capital Loss']; del one_hot_dat['Hours per week']; del one_hot_dat['Country']; del one_hot_dat['Target']
#del one_hot_dat['>50K']
one_hot_dat.head()


# # Challenge Problem: Alternative Encoding Scheme to One-Hot Encoding

# Likewise, beside doing a One-hot encoding, we could also map each string label in our categorical features to a integral value. As we previously leveraged a Pandas data frame method to do the encoding, we are now going to test out a scikit-learn method to impose the integral value encoding. Please check the docs and read up on: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html. Proceed with the encoding and build a Naive Bayes and Logistic classifier for both. Do we get similar results? What should we expect? And why?

# In[8]:

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

the_labelizer = preprocessing.LabelEncoder()
adult_dat_2 = adult_dat

def integral_encoding(table):
    the_labelizer = {}
    
    for col in table.columns:
        if table.dtypes[col] == np.object:
            the_labelizer[col] = preprocessing.LabelEncoder()
            table[col] = the_labelizer[col].fit_transform(table[col])
            
    return table

integral_encoding(adult_dat_2).head()


# # Summarize the data and engage in elementary data exploration

# For some data exploration, use Pandas histogram methods to display the features. 

# In[125]:

# Write histogram functions here, and/or any other data visualizations


# # Partition the data

# Without using any direct method/libraries that would automatically accomplish this, please partition the data set 70/30. You can use anything from the math, pandas, or numpy library, do not use other libraries. 

# In[75]:

partition_val = np.random.rand(len(one_hot_dat)) < 0.70
train = one_hot_dat[partition_val]
test = one_hot_dat[~partition_val]


# # Define your feature set and define your target 

# In[85]:

target_train = train['<=50K']
feature_train = train.drop('<=50K', axis=1)


# # Run Naive Bayes Classifier

# Instantiate the Naive Bayes predictor from scikit-learn with the training data. 

# In[90]:

Cat_Naive_Bayes = naive_bayes.MultinomialNB();
Cat_Naive_Bayes.fit(feature_train, target_train)


# # Check Accuracy / Score for Naive Bayes

# Define the target and feature set for the test data

# In[87]:

target_test = test['<=50K']
feature_test =  test.drop('<=50K', axis = 1)


# Score the Naive Bayes classifier on the test data

# In[88]:

Cat_Naive_Bayes.score(feature_test, target_test)


# # Check Accuracy / Score for a Logistic Classifier 

# Define a logistic regression and train it with the feature and target set

# In[118]:

import sklearn.linear_model as linear_model

logistic_class = linear_model.LogisticRegression()
logit = logistic_class.fit(feature_train, target_train)


# Produce the accuracy score of the logistic regression from the test set

# In[119]:

logit.score(feature_test, target_test)


# Was that what you expected? All we did was remove non categorical variables, and imposed a One-hot encoding, should we have expected the Naive Bayes to underperform the Logistic? Here are some other things you can think about:
# 
# 1. What other metrics outside of simple accuracy can we utilize to measure performance?
# 2. Could some pair-wise correlation between pair-wise features in our feature set have caused an issue with the Naive Bayes? What are the assumptions for Naive Bayes which may cause this to happen? 
# 3. How could we improve the performance of Naive Bayes? 
# 4. What about the numerica features we left out, should we bring them back in? How?
# 
# If you want to expand on your analysis, why not build a correlation matrix, or perhaps print a summary of the logistic regression, would an ANOVA table help in our assessment for this case? 

# In[ ]:



