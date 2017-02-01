
# coding: utf-8

# In[51]:

import foursquare
import pandas as pd


# In[26]:

CLIENT_ID = 'AELYR2TEGMVFRL4ANU51B3ECDIDX1ZTJVLXLKXJJBKS0YQTJ'   # Input your client id/ client secret here
CLIENT_SECRET = 'JXRHBAFSFGCSKP1UUKNJYKWPUAPI131XIICXUWMJNCWPJRYC'
client = foursquare.Foursquare(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)


# In[27]:

bounding_box = [-71.0600316525,42.3507938209,-71.0549998283,42.3544490803]


# In[53]:

venue_search = client.venues.search(params={'ll': '42.3544490803,-71.0549998283'})


# In[54]:

venue_search[]


# In[ ]:



