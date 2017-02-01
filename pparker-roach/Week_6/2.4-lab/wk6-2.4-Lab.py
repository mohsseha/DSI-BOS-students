
# coding: utf-8

# # Decision Trees and Ensembles Lab
# 
# In this lab we will compare the performance of a simple Decision Tree classifier with a Bagging classifier. We will do that on few datasets, starting from the ones offered by Scikit Learn.

# ## 1. Breast Cancer Dataset
# We will start our comparison on the breast cancer dataset.
# You can load it directly from scikit-learn using the `load_breast_cancer` function.
# 
# ### 1.a Simple comparison
# 1. Load the data and create X and y
# - Initialize a Decision Tree Classifier and use cross_val_score to evaluate it's performance. Set crossvalidation to 5-folds
# - Wrap a Bagging Classifier around the Decision Tree Classifier and use cross_val_score to evaluate it's performance. Set crossvalidation to 5-folds. 
# - Which score is better? Are the score significantly different? How can you judge that?

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[ ]:




# ### 1.b Scaled pipelines
# As you may have noticed the features are not normalized. Do the score improve with normalization?
# By now you should be very familiar with pipelines and scaling, so:
# 
# 1. Create 2 pipelines, with a scaling preprocessing step and then either a decision tree or a bagging decision tree.
# - Which score is better? Are the score significantly different? How can you judge that?
# - Are the scores different from the non-scaled data?

# In[ ]:




# ### 1.c Grid Search
# 
# Grid search is a great way to improve the performance of a classifier. Let's explore the parameter space of both models and see if we can improve their performance.
# 
# 1. Initialize a GridSearchCV with 5-fold cross validation for the Decision Tree Classifier
# - search for few values of the parameters in order to improve the score of the classifier
# - Use the whole X, y dataset for your test
# - Check the best\_score\_ once you've trained it. Is it better than before?
# - How does the score of the Grid-searched DT compare with the score of the Bagging DT?
# - Initialize a GridSearchCV with 5-fold cross validation for the Bagging Decision Tree Classifier
# - Repeat the search
#     - Note that you'll have to change parameter names for the base_estimator
#     - Note that there are also additional parameters to change
#     - Note that you may end up with a grid space to large to search in a short time
#     - Make use of the n_jobs parameter to speed up your grid search
# - Does the score improve for the Grid-searched Bagging Classifier?
# - Which score is better? Are the score significantly different? How can you judge that?

# In[ ]:




# ## 2 Diabetes and Regression
# 
# Scikit Learn has a dataset of diabetic patients obtained from this study:
# 
# http://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
# http://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf
# 
# 442 diabetes patients were measured on 10 baseline variables: age, sex, body mass index, average blood pressure, and six blood serum measurements.
# 
# The target is a quantitative measure of disease progression one year after baseline.
# 
# Repeat the above comparison between a DecisionTreeRegressor and a Bagging version of the same.

# ### 2.a Simple comparison
# 1. Load the data and create X and y
# - Initialize a Decision Tree Regressor and use cross_val_score to evaluate it's performance. Set crossvalidation to 5-folds. Which score will you use?
# - Wrap a Bagging Regressor around the Decision Tree Regressor and use cross_val_score to evaluate it's performance. Set crossvalidation to 5-folds. 
# - Which score is better? Are the score significantly different? How can you judge that?

# In[ ]:




# ### 2.b Grid Search
# 
# Repeat Grid search as above:
# 
# 1. Initialize a GridSearchCV with 5-fold cross validation for the Decision Tree Regressor
# - Search for few values of the parameters in order to improve the score of the regressor
# - Use the whole X, y dataset for your test
# - Check the best\_score\_ once you've trained it. Is it better than before?
# - How does the score of the Grid-searched DT compare with the score of the Bagging DT?
# - Initialize a GridSearchCV with 5-fold cross validation for the Bagging Decision Tree Regressor
# - Repeat the search
#     - Note that you'll have to change parameter names for the base_estimator
#     - Note that there are also additional parameters to change
#     - Note that you may end up with a grid space to large to search in a short time
#     - Make use of the n_jobs parameter to speed up your grid search
# - Does the score improve for the Grid-searched Bagging Regressor?
# - Which score is better? Are the score significantly different? How can you judge that?
# 

# In[ ]:




# ## Bonus: Project 6 data
# 
# Repeat the analysis for the Project 6 Dataset

# In[ ]:



