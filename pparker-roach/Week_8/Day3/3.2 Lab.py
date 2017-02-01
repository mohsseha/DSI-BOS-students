
# coding: utf-8

# In[1]:

import foursquare
import json
import pandas as pd
import unicodedata


# In[2]:

CLIENT_ID = 'AELYR2TEGMVFRL4ANU51B3ECDIDX1ZTJVLXLKXJJBKS0YQTJ'   # Input your client id/ client secret here
CLIENT_SECRET = 'JXRHBAFSFGCSKP1UUKNJYKWPUAPI131XIICXUWMJNCWPJRYC'
client = foursquare.Foursquare(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)


# In[3]:

bounding_box = [-71.0600316525,42.3507938209,-71.0549998283,42.3544490803]


# In[7]:

venue_search = client.venues.search(params={'ll': '42.3544490803,-71.0549998283'})
len(venue_search['venues'])


# In[73]:

print(json.dumps(venue_search, indent=5))


# In[78]:

venue_search['venues'][0]['categories'][0]['shortName']


# In[79]:

len(venue_search['venues'])


# In[82]:

venue_search['venues'][2]['beenHere']


# In[95]:

#temp = unicodedata.normalize('NFKD', venue_search['venues'][17]['id']).encode('ascii','ignore')
temp = venue_search['venues'][17]['id']


# In[96]:

temp1 = client.venues(temp);


# In[89]:

temp1


# In[90]:

map(lambda h: h['text'], temp1['venue']['tips']['groups'][0]['items'])


# In[101]:

print(json.dumps(venue_search))


# In[ ]:



