
# coding: utf-8

# In[27]:

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
# - These are websites that always relevant like recipes or reviews (as opposed to current events)
# - Look at some examples

# In[28]:

data = pd.read_csv('C:/Users/Pat.NOAGALLERY/Documents/data_sources//train.tsv', sep='\t', na_values={'is_news' : '?'}).fillna(0)

# Extract the title and body from the boilerplate JSON text
data['title'] = data.boilerplate.map(lambda x: json.loads(x).get('title', ''))
data['body'] = data.boilerplate.map(lambda x: json.loads(x).get('body', ''))


# In[41]:

data[['title', 'label']].head()


# #### Does being a news site effect green-ness?

# In[30]:

print(data.groupby('is_news')[['label']].mean())
sb.factorplot(x='is_news', y='label', data=data, kind='bar')


# #### Does the website category effect green-ness?

# In[31]:

print(data.groupby('alchemy_category')[['label']].mean())
cat_plot = sb.factorplot(x='alchemy_category', y='label', data=data, kind='bar')
cat_plot.set_xticklabels(rotation=90)


# #### Does the image ratio effect green-ness?

# In[32]:

#print(data.groupby('image_ratio')[['label']].mean())
sb.factorplot(x='image_ratio', y='label', data=data, kind='bar')


# #### Fit a logistic regression model using statsmodels
# - Test different features that may be valuable
# - Examine the coefficients, does the feature increase or decrease the effect of being evergreen?

# In[47]:

import statsmodels.formula.api as sm

model = sm.logit("label ~ alchemy_category",data = data).fit()
model.summary() 


# In[ ]:




# #### Fit a logistic regression model using statsmodels with text features
# - Add text features that may be useful, add this to the model and see if they improve the fit
# - Examine the coefficients, does the feature increase or decrease the effect of being evergreen?

# In[52]:

# EXAMPLE text feature 'recipe'
data['is_recipe'] = data['title'].fillna('').str.contains('recipe')
model = sm.logit("label ~ is_recipe",data = data).fit()
model.summary() 


# In[54]:

data['alchemy_category'] = data['title'].fillna('').str.contains('business')
model = sm.logit("label ~ is_recipe + alchemy_category",data = data).fit()
model.summary() 


# In[ ]:



