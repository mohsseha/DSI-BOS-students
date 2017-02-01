
# coding: utf-8

# In[2]:

import foursquare
import pandas


# In[7]:

CLIENT_ID = '393786618'   # Input your client id/ client secret here
CLIENT_SECRET = 'Arrghh11'
client = foursquare.Foursquare(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)


# In[8]:

bounding_box = [-71.0600316525,42.3507938209,-71.0549998283,42.3544490803]


# In[6]:

client.venues.search(params={'ll': '-71.0600316525,42.3507938209'})  


# In[ ]:



