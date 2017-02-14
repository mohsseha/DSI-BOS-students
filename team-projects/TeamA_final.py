
# coding: utf-8

# In[1]:

import pandas as pd
from sklearn import linear_model


# In[4]:

class TeamAPredictor():
    def __init__(self):
        self.training_tickers = ["TMO","AMZN","AAPL","GM","IBM","TWTR","CBS","TM","BIO"]

        
    def get_data(self, ticker):
        import requests
        import pandas as pd
        import io
        import numpy as np

        # parameters to modify in the query
        payload = {
        'output':'csv',
       'startdate':'2000-01-01',
       'enddate':'2017-01-01',   
       'q':ticker
       }

        URL = 'http://www.google.com/finance/historical' # base URL

        r = requests.get(URL, params=payload).content # create the requests object
        df = pd.read_csv(io.StringIO(r.decode('utf-8')))
        # dropping unnecessary columns and setting the index as datetime
        df = df.drop(['Open', 'High', 'Low', 'Volume'], axis=1)
        df.set_index('Date', inplace=True)
        df.index = df.index.to_datetime()

        # DataFrame manipulation
        offset_1, offset_2, offset_3,offset_4  = int(252*3), int(252*3.25), int(252*3.5), int(252*3.75) # approximately one year, two years and three years
        df_length = len(df) - offset_4 # determines the maximum length possible for our dataframe

        x_4 = df[:df_length] # target variable
        x_3 = df[offset_1:df_length + offset_1]  
        x_2 = df[offset_2:df_length + offset_2] 
        x_1 = df[offset_3:df_length + offset_3] 
        x_0 = df[offset_4: df_length + offset_4] 

        # modify the indexes so we can concatenate the dataframes
        x_4 = x_4.reset_index().drop('index', axis=1)
        x_3 = x_3.reset_index().drop('index', axis=1)
        x_2 = x_2.reset_index().drop('index', axis=1)
        x_1 = x_1.reset_index().drop('index', axis=1)
        x_0 = x_0.reset_index().drop('index', axis=1)
 
        frames = [x_0, x_1, x_2, x_3, x_4] # list of dataframes to pass into pd.concat

        # concat the dataframes and rename the columns
        final_df = pd.concat(frames, axis=1)
        final_df.columns = ['x_0', 'x_1', 'x_2', 'x_3','y']

        return final_df
    
    def predict_2017_1_1_price(self, ticker):
    
        df=self.get_data(ticker)
        X = df[['x_0', 'x_1', 'x_2', 'x_3']] 
        y = df['y'] 
        lm_LassoCV = linear_model.LassoCV(alphas=[0.01,0.1, 1, 10,100])
        model_LassoCV = lm_LassoCV.fit(X, y)
        predictions_LassoCV = model_LassoCV.predict(X)
        #plt.scatter(y, predictions_LassoCV)
    
        return X['x_0'][0], predictions_LassoCV[0]    #returns 2014 price and 2017 predicted price
    
    def predict_from_2014_to_2017(self, stocks,investment):
        predicted_2017_investment = 0
        for ticker in stocks:
            x_0, y = self.predict_2017_1_1_price(ticker)
            predicted_2017_investment +=(y/float(x_0)/len(stocks)*investment) 

        return predicted_2017_investment


# In[5]:

pr = TeamAPredictor()


# In[6]:

pr.predict_from_2014_to_2017(["VIAB","GOOG","F","MSFT"], 10000)


# In[ ]:



