
# coding: utf-8

# In[ ]:

from sqlalchemy import create_engine


# In[ ]:

engine = create_engine('postgresql://postgres:root@localhost:5432/test')

data.to_sql('movies', engine)

