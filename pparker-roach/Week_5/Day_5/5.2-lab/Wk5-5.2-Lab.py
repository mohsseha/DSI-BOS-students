
# coding: utf-8

# # Support Vector Machines Lab

# In this lab we will explore several datasets with SVMs. The assets folder contains several datasets (in order of complexity):
# 
# 1. Breast cancer
# - Spambase
# - Car evaluation
# - Mushroom
# 
# For each of these a `.names` file is provided with details on the origin of data.

# In[66]:

import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


# # Exercise 1: Breast Cancer
# 
# 
# 
# ## 1.a: Load the Data
# Use `pandas.read_csv` to load the data and assess the following:
# - Are there any missing values? (how are they encoded? do we impute them?)
# - Are the features categorical or numerical?
# - Are the values normalized?
# - How many classes are there in the target?
# 
# Perform what's necessary to get to a point where you have a feature matrix `X` and a target vector `y`, both with only numerical entries.
# 
# C:/Users/Pat.NOAGALLERY/Documents/data_sources/

# In[67]:

bc = pd.read_csv("C:/Users/Pat.NOAGALLERY/Documents/data_sources/breast_cancer.csv")
bc.info()


# In[68]:

bc.describe()


# In[69]:

bc.head()


# In[70]:

for col in bc.columns:
    print(col ," ", bc[col].dtype)
print(bc[bc['Bare_Nuclei']=='?'].count())

#since there are only 16 rows where there are question marks I am deleting those rows from the dataset
bc = bc[bc.Bare_Nuclei != "?"]
bc.shape


# ## 1.b: Model Building
# 
# - What's the baseline for the accuracy?
# - Initialize and train a linear svm. What's the average accuracy score with a 3-fold cross validation?
# - Repeat using an rbf classifier. Compare the scores. Which one is better?
# - Are your features normalized? if not, try normalizing and repeat the test. Does the score improve?
# - What's the best model?
# - Print a confusion matrix and classification report for your best model using:
#         train_test_split(X, y, stratify=y, test_size=0.33, random_state=42)
# 
# **Check** to decide which model is best, look at the average cross validation score. Are the scores significantly different from one another?

# In[71]:

X = bc.drop(['Sample_code_number', 'Class'], axis = 1)
y = bc['Class'] == 4
y.shape


# In[72]:

print (y.value_counts())
#baseline => 65%
y.value_counts() / len(y)


# In[73]:

from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score

all_scores = []
model = SVC(kernel='linear')

def do_cv(model, X, y, cv):
    scores = cross_val_score(model, X, y, cv=cv)
    print (model)
    sm = scores.mean()
    ss = scores.std()
    res = (sm, ss)
    print ("Average score: {:0.3}+/-{:0.3}".format(*res))
    return res

all_scores.append(do_cv(model, X, y, 3))



# In[74]:

model = SVC(kernel='rbf')
all_scores.append(do_cv(model, X, y, 3))


# In[75]:

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
all_scores.append(do_cv(model, X, y, 3))


# In[76]:

model = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
all_scores.append(do_cv(model, X, y, 3))


# In[78]:

from sklearn.metrics import confusion_matrix, classification_report

def print_cm_cr(y_true, y_pred, names):
    """prints the confusion matrix and the classification report"""
    cm = confusion_matrix(y_true, y_pred)
    cols = ['pred_' + c for c in names]
    dfcm = pd.DataFrame(cm, columns = cols, index = names)
    print (dfcm)
    print ()
    print (classification_report(y_true, y_pred))


# In[79]:

from sklearn.cross_validation import train_test_split


# In[80]:

model = make_pipeline(StandardScaler(), SVC(kernel='rbf'))


# In[ ]:




# **Check:** Are there more false positives or false negatives? Is this good or bad?

# ## 1.c: Feature Selection
# 
# Use any of the strategies offered by `sklearn` to select the most important features.
# 
# Repeat the cross validation with only those 5 features. Does the score change?

# In[19]:




# ## 1.d: Learning Curves
# 
# Learning curves are useful to study the behavior of training and test errors as a function of the number of datapoints available.
# 
# - Plot learning curves for train sizes between 10% and 100% (use StratifiedKFold with 5 folds as cross validation)
# - What can you say about the dataset? do you need more data or do you need a better model?

# In[ ]:




# ##  1.e: Grid Ssearch
# 
# Use the grid_search function to explore different kernels and values for the C parameter.
# 
# - Can you improve on your best previous score?
# - Print the best parameters and the best score

# In[23]:




# # Exercise 2
# Now that you've completed steps 1.a through 1.e it's time to tackle some harder datasets. But before we do that, let's encapsulate a few things into functions so that it's easier to repeat the analysis.
# 
# ## 2.a: Cross Validation
# Implement a function `do_cv(model, X, y, cv)` that does the following:
# - Calculates the cross validation scores
# - Prints the model
# - Prints and returns the mean and the standard deviation of the cross validation scores
# 
# > Answer: see above
# 
# ## 2.b: Confusion Matrix and Classification report
# Implement a function `do_cm_cr(model, X, y, names)` that automates the following:
# - Split the data using `train_test_split(X, y, stratify=y, test_size=0.33, random_state=42)`
# - Fit the model
# - Prints confusion matrix and classification report in a nice format
# 
# **Hint:** names is the list of target classes
# 
# > Answer: see above
# 
# ## 2.c: Learning Curves
# Implement a function `do_learning_curve(model, X, y, sizes)` that automates drawing the learning curves:
# - Allow for sizes input
# - Use 5-fold StratifiedKFold cross validation
# 
# > Answer: see above
# 
# ## 2.d: Grid Search
# Implement a function `do_grid_search(model, parameters)` that automates the grid search by doing:
# - Calculate grid search
# - Print best parameters
# - Print best score
# - Return best estimator
# 
# 
# > Answer: see above

# # Exercise 3
# Using the functions above, analyze the Spambase dataset.
# 
# Notice that now you have many more features. Focus your attention on step C => feature selection
# 
# - Load the data and get to X, y
# - Select the 15 best features
# - Perform grid search to determine best model
# - Display learning curves

# In[ ]:




# # Exercise 4
# Repeat steps 1.a - 1.e for the car dataset. Notice that now features are categorical, not numerical.
# - Find a suitable way to encode them
# - How does this change our modeling strategy?
# 
# Also notice that the target variable `acceptability` has 4 classes. How do we encode them?
# 

# In[ ]:




# # Bonus
# Repeat steps 1.a - 1.e for the mushroom dataset. Notice that now features are categorical, not numerical. This dataset is quite large.
# - How does this change our modeling strategy?
# - Can we use feature selection to improve this?
# 

# In[ ]:



