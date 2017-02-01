
# coding: utf-8

# In[35]:

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine


# In[30]:

engine = create_engine('postgresql://postgres:root@localhost:5432/dummy')


# In[31]:

test_dict = {"name": "char(18)","age": 'int'}


# In[32]:

test_df = pd.DataFrame(test_dict,index=["dtype"])
test_df = test_df.T
test_df


# In[33]:


test_rows = pd.DataFrame({'name': ["bubba", "barack", "michele"], 'age': [44, 54, 33]})
test_rows


# In[ ]:




# In[39]:

test_rows.to_sql('age',engine, 
                      if_exists = 'append', 
                      index = False, 
)


# In[2]:

from geopy.distance import vincenty
newport_ri = (41.49008, -71.312796)
cleveland_oh = (41.499498, -81.695391)
print(vincenty(newport_ri, cleveland_oh).miles)
#538.3904451566326


# In[ ]:



