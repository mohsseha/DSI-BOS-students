
# coding: utf-8

# # Natural Language Processing Lab
# 
# In this lab we will further explore Scikit's and NLTK's capabilities to process text. We will use the 20 Newsgroup dataset, which is provided by Scikit-Learn.

# In[19]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[49]:

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# In[50]:


categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=('headers', 'footers', 'quotes'))

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=('headers', 'footers', 'quotes'))


# ## 1. Data inspection
# 
# We have downloaded a few newsgroup categories and removed headers, footers and quotes.
# 
# Let's inspect them.
# 
# 1. What data taype is `data_train`
# > sklearn.datasets.base.Bunch
# - Is it like a list? Or like a Dictionary? or what?
# > Dict
# - How many data points does it contain?
# - Inspect the first data point, what does it look like?
# > A blurb of text

# In[51]:

print(data_train["description"])
print(data_train.keys())
print(data_train['data'][1])
print(data_train["filenames"])


# ## 2. Bag of Words model
# 
# Let's train a model using a simple count vectorizer
# 
# 1. Initialize a standard CountVectorizer and fit the training data
# - how big is the feature dictionary
# - repeat eliminating english stop words
# - is the dictionary smaller?
# - transform the training data using the trained vectorizer
# - what are the 20 words that are most common in the whole corpus?
# - what are the 20 most common words in each of the 4 classes?
# - evaluate the performance of a Lotistic Regression on the features extracted by the CountVectorizer
#     - you will have to transform the test_set too. Be carefule to use the trained vectorizer, without re-fitting it
# - try the following 3 modification:
#     - restrict the max_features
#     - change max_df and min_df
#     - use a fixed vocabulary of size 80 combining the 20 most common words per group found earlier
# - for each of the above print a confusion matrix and investigate what gets mixed
# > Anwer: not surprisingly if we reduce the feature space we lose accuracy
# - print out the number of features for each model

# In[53]:

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# In[56]:

count_vec = CountVectorizer()
count_vec.fit(data_train['data'])


# In[58]:

len(count_vec.get_feature_names())


# In[59]:

count_vec = CountVectorizer(stop_words='english')
count_vec.fit(data_train['data'])


# In[65]:

len(count_vec.get_feature_names())
#count_vec.get_feature_names()


# In[73]:

X_train = pd.DataFrame(count_vec.transform(data_train['data']).todense(),
                       columns=count_vec.get_feature_names())
print(X_train.shape)
#X_train


# In[85]:

word_counts = X_train.sum(axis=0)
print("all class word count", word_counts.sort_values(ascending = False).head(20))
y_train = data_train['target']
common_words = []
# for i in xrange(4):
#     word_count = X_train[y_train==i].sum(axis=0)
#     print (names[i], "most common words")
#     cw = word_count.sort_values(ascending = False).head(20)
# #     cw.to_csv('../../../5.2-lesson/assets/datasets/'+names[i]+'_most_common_words.csv')
#     print (cw)
#     common_words.extend(cw.index)
#     print ()
    


# ## 3. Hashing and TF-IDF
# 
# Let's see if Hashing or TF-IDF improves the accuracy.
# 
# 1. Initialize a HashingVectorizer and repeat the test with no restriction on the number of features
# - does the score improve with respect to the count vectorizer?
#     - can you change any of the default parameters to improve it?
# - print out the number of features for this model
# - Initialize a TF-IDF Vectorizer and repeat the analysis above
# - can you improve on your best score above?
#     - can you change any of the default parameters to improve it?
# - print out the number of features for this model

# In[ ]:




# In[ ]:




# ## 4. Classifier comparison
# 
# Of all the vectorizers tested above, choose one that has a reasonable performance with a manageable number of features and compare the performance of these models:
# 
# - KNN
# - Logistic Regression
# - Decision Trees
# - Support Vector Machine
# - Random Forest
# - Extra Trees
# 
# In order to speed up the calculation it's better to vectorize the data only once and then compare the models.

# In[ ]:




# ## Bonus: Other classifiers
# 
# Adapt the code from [this example](http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#example-text-document-classification-20newsgroups-py) to compare across all the classifiers suggested and to display the final plot

# In[ ]:




# ## Bonus: NLTK
# 
# NLTK is a vast library. Can you find some interesting bits to share with classmates?
# Start here: http://www.nltk.org/
