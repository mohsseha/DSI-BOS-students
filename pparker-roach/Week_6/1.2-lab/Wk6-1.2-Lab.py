
# coding: utf-8

# # Decision Trees Lab
# 
# In this lab we will discover how to apply decision trees to regression and classification problems.

# ### 1: Build a regression tree
# 
# How do you build a decision tree? You're going to find out by building one in pairs!
# 
# Your training data is a tiny dataset of [used vehicle sale prices](../../assets/datasets/used_cars.csv). Your goal is to predict Price for out-of-sample data. Here are your instructions:
# 
# 1. Read the data into Pandas.
# - Explore the data by sorting, plotting, or split-apply-combine (aka `group_by`).
# - Decide which feature is the most important predictor, and use that to make your first split. (Only binary splits are allowed!)
# - After making your first split, you should actually split your data in Pandas into two parts, and then explore each part to figure out what other splits to make.
# - Decide if you need additional splits along other features
# - Stop making splits once you are convinced that it strikes a good balance between underfitting and overfitting. (As always, your goal is to build a model that generalizes well!)
# - You are allowed to split on the same variable multiple times!
# - Draw your tree on a piece of paper, making sure to label your leaves with the mean Price for the observations in that "bucket".
# - When you're finished, review your tree to make sure nothing is backwards. (Remember: follow the left branch if the rule is true, and follow the right branch if the rule is false.)

# In[32]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[24]:

# 1. read the data into Pandas
data = pd.read_csv("C:/Users/Pat.NOAGALLERY/Documents/data_sources/used_cars.csv")
print(data.shape)
print(data.info())
data.describe()
data['price_range'] = pd.qcut(data['price'],2,labels=["low","high"])

data['age_range'] = pd.qcut(2017-data['year'],2,labels=["newer", "older"])
data['milage'] = pd.qcut(data['miles'],2,labels=["low_milage", "high_milage"])
data


# In[39]:

data[['price','type']].plot(kind='hist')
plt.show()


# #### How does a computer build a regression tree?
# 
# The ideal approach would be for the computer to consider every possible partition of the feature space. However, this is computationally infeasible, so instead an approach is used called **recursive binary splitting:**
# 
# - Begin at the top of the tree.
# - For every single predictor, examine every possible cutpoint, and choose the predictor and cutpoint such that the resulting tree has the **lowest possible mean squared error (MSE)**. Make that split.
# - Repeat the examination for the two resulting regions, and again make a single split (in one of the regions) to minimize the MSE.
# - Keep repeating this process until a stopping criteria is met.
# 
# **How does it know when to stop?**
# 
# 1. We could define a stopping criterion, such as a **maximum depth** of the tree or the **minimum number of samples in the leaf**.
# 2. We could grow the tree deep, and then "prune" it back using a method such as "cost complexity pruning" (aka "weakest link pruning").
# 
# Method 2 involves setting a tuning parameter that penalizes the tree for having too many leaves. As the parameter is increased, branches automatically get pruned from the tree, resulting in smaller and smaller trees. The tuning parameter can be selected through cross-validation.
# 
# Note: **Method 2 is not currently supported by scikit-learn**, and so we will use Method 1 instead.
# 

# ### 2: Build a regression tree in scikit-learn
# 
# Building a tree by hand was not so easy, and also not ideal. Let's use scikit-learn to build an optimal regression tree. Do the following:
# 
# - Map the `type` column to a binary variable
# - Create a matrix `X` that contains the feature values and a vector `y` that contains the price values
# - Split the data into train-test using a random state of 42 and test_size of 30%
# - Import and initialize the `DecisionTreeRegressor` class from scikit-learn
# - Fit it to the training set
# - Predict the values of the test set
# - Display the predicted and actual values in a plot
# - Use r2_score to judge the goodness of the regression

# In[41]:

# convert car to 0 and truck to 1
data['type'] = data.type.map({'car':0, 'truck':1})
data


# ### 3.b Global parameters
# 
# The `DecisionTreeRegressor` offers few global parameters that can be changed at initialization. For example one can set the `max_depth` or the `min_samples_leaf` parameters and impose global constraints on the space of solutions.
# 
# 1. Use `cross_val_score` with 3-fold cross validation to find the optimal value for the `max_depth` (explore values 1 - 10). Note that you will have to set `scoring='mean_squared_error'` as criterion for score. Always set `random_state=1`
# - Plot the error as a function of `max_depth`

# In[ ]:




# ## 3.c Feature importances
# 
# The decision tree class exposes an attribute called `feature_importances_`.
# 
# 1. Check the importance of each feature. what's the most important feature?

# In[ ]:




# ### 3.d Tree visualization
# 
# Follow the example in the [documentation](http://scikit-learn.org/stable/modules/tree.html) to visualize the tree.
# You may have to install `pydot` and/or `graphviz` if you don't have them already.

# In[ ]:




# #### Interpreting a tree diagram
# 
# How do we read this decision tree?
# 
# **Internal nodes:**
# 
# - `samples` is the number of observations in that node before splitting
# - `mse` is the mean squared error calculated by comparing the actual response values in that node against the mean response value in that node
# - First line is the condition used to split that node (go left if true, go right if false)
# 
# **Leaves:**
# 
# - `samples` is the number of observations in that node
# - `value` is the mean response value in that node
# - `mse` is the mean squared error calculated by comparing the actual response values in that node against "value"

# ### Exercise 4: Use GridSearchCV to find te best Regression Tree
# 
# How do we know by pruning with max depth is the best model for us? Trees offer a variety of ways to pre-prune (that is, we tell a computer how to design the resulting tree with certain "gotchas").
# 
# Measure           | What it does
# ------------------|-------------
# max_depth         | How many nodes deep can the decision tree go?
# max_features      | Is there a cut off to the number of features to use?
# max_leaf_nodes    | How many leaves can be generated per node?
# min_samples_leaf  | How many samples need to be included at a leaf, at a minimum?  
# min_samples_split | How many samples need to be included at a node, at a minimum?
# 
# 1. Initialize reasonable ranges for all parameters and find the optimal combination using Grid Search.

# In[ ]:




# ## 4 Classification trees
# 
# Classification trees are very similar to regression trees. Here is a quick comparison:
# 
# |regression trees|classification trees|
# |---|---|
# |predict a continuous response|predict a categorical response|
# |predict using mean response of each leaf|predict using most commonly occuring class of each leaf|
# |splits are chosen to minimize MSE|splits are chosen to minimize a different criterion (discussed below)|
# 
# Note that classification trees easily handle **more than two response classes**! (How have other classification models we've seen handled this scenario?)
# 
# Here's an **example of a classification tree**, which predicts whether or not a patient who presented with chest pain has heart disease:

# ### 4.a Building a classification tree in scikit-learn
# We'll build a classification tree using the [Car Dataset](./assets/datasets/cars.csv).
# 
# - Load the dataset in pandas
# - Check for missing values
# - Encode all the categorical features to booleans using `pd.get_dummies`
# - Encode the labels using LabelEncoder
# - Split X and y with train_test split like above
#         train_test_split(X, y, test_size=0.3, random_state=42)
# - Fit a classification tree with `max_depth=3` on all data
# - Visualize the tree using graphviz
# - Compute the feature importances
# - Compute and display the confusion matrix
# - Release the constraint of `max_depth=3` and see if the classification improves

# ## Bonus
# 
# Visualize the last tree. Can you make sense of it? What does this teach you about decision tree interpretability?
# 

# In[ ]:



