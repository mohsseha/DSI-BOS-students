
# coding: utf-8

# In[1]:

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import json
get_ipython().magic('matplotlib inline')

pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 300


# ## Predicting "Greenness" Of Content
# 
# This dataset comes from [stumbleupon](https://www.stumbleupon.com/), a web page recommender and was made available [here](https://www.kaggle.com/c/stumbleupon/download/train.tsv)
# 
# A description of the columns is below
# 
# FieldName|Type|Description
# ---------|----|-----------
# url|string|Url of the webpage to be classified
# urlid|integer| StumbleUpon's unique identifier for each url
# boilerplate|json|Boilerplate text
# alchemy_category|string|Alchemy category (per the publicly available Alchemy API found at www.alchemyapi.com)
# alchemy_category_score|double|Alchemy category score (per the publicly available Alchemy API found at www.alchemyapi.com)
# avglinksize| double|Average number of words in each link
# commonLinkRatio_1|double|# of links sharing at least 1 word with 1 other links / # of links
# commonLinkRatio_2|double|# of links sharing at least 1 word with 2 other links / # of links
# commonLinkRatio_3|double|# of links sharing at least 1 word with 3 other links / # of links
# commonLinkRatio_4|double|# of links sharing at least 1 word with 4 other links / # of links
# compression_ratio|double|Compression achieved on this page via gzip (measure of redundancy)
# embed_ratio|double|Count of number of <embed> usage
# frameBased|integer (0 or 1)|A page is frame-based (1) if it has no body markup but have a frameset markup
# frameTagRatio|double|Ratio of iframe markups over total number of markups
# hasDomainLink|integer (0 or 1)|True (1) if it contains an <a> with an url with domain
# html_ratio|double|Ratio of tags vs text in the page
# image_ratio|double|Ratio of <img> tags vs text in the page
# is_news|integer (0 or 1) | True (1) if StumbleUpon's news classifier determines that this webpage is news
# lengthyLinkDomain| integer (0 or 1)|True (1) if at least 3 <a> 's text contains more than 30 alphanumeric characters
# linkwordscore|double|Percentage of words on the page that are in hyperlink's text
# news_front_page| integer (0 or 1)|True (1) if StumbleUpon's news classifier determines that this webpage is front-page news
# non_markup_alphanum_characters|integer| Page's text's number of alphanumeric characters
# numberOfLinks|integer Number of <a>|markups
# numwords_in_url| double|Number of words in url
# parametrizedLinkRatio|double|A link is parametrized if it's url contains parameters or has an attached onClick event
# spelling_errors_ratio|double|Ratio of words not found in wiki (considered to be a spelling mistake)
# label|integer (0 or 1)|User-determined label. Either evergreen (1) or non-evergreen (0); available for train.tsv only

# ### What are 'evergreen' sites?
# - These are websites that always relevant like recipies or reviews (as opposed to current events)
# - Look at some examples

# In[2]:

data = pd.read_csv('../../assets/datasets/train.tsv', sep='\t', na_values='?')

# Extract the title and body from the boilerplate JSON text
data['title'] = data.boilerplate.map(lambda x: json.loads(x).get('title', ''))
data['body'] = data.boilerplate.map(lambda x: json.loads(x).get('body', ''))


# In[3]:

data[['title', 'label']].head()


# #### Build a feature matrix where X contains useful features and y contains the target variable `label`

# In[20]:

X = data[[
        'linkwordscore',
        'news_front_page',
        'is_news',
        'html_ratio',
        'image_ratio'
    ]].fillna(0)

y = data.label


# #### Build a Logistic Regression model using scikit-learn
# - Examine the coefficients using the `examine_coefficients` function provided
# - Evaluate the AUC, precision and recall of the model using cross-validation
# - Plot the ROC curve
# - Iterate on the model by adding in additional features in `X` above

# In[5]:

def examine_coefficients(model, df):
    return pd.DataFrame(
        { 'Coefficient' : model.coef_[0] , 'Feature' : df.columns}
    ).sort_values(by='Coefficient')


# In[6]:

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
# REMEMBER: by default this model has regularization which we will revisit at the end of the week


# In[ ]:

model.fit(X, y) # This fits the model to learn the coefficients

examine_coefficients(model, X)


# In[ ]:

from sklearn.cross_validation import cross_val_score

for metric in ['accuracy', 'precision', 'recall', 'roc_auc']:
    scores = cross_val_score(model, X, y, scoring=metric)
    print("mean {}: {}, all: {}".format(metric, scores.mean(), scores))


# In[ ]:



