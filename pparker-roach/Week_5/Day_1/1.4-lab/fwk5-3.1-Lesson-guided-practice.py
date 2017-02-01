
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
engine = create_engine('postgresql://dsi_student:gastudents@dsi.c20gkj5cvu3l.us-east-1.rds.amazonaws.com/titanic')

df = pd.read_sql('SELECT * FROM train', engine)

