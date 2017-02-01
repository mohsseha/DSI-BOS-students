
# coding: utf-8

# In[16]:

# import all depedencies first cell
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics, cross_validation


# In[2]:

# define functions in one cell, call when you need
def read_csv(path):
    return pd.read_csv(path)


# In[3]:

path = 'C:/Users/Pat.NOAGALLERY/Documents/data_sources/bikeshare.csv'
bikeshare = read_csv(path)


# In[15]:

# get the data dictionary of bikeshare to see possible feature candidate
print(bikeshare.shape)
print(bikeshare.columns)
print(bikeshare.dtypes)
print(bikeshare.head(3))


# 
# - instant: record index
# - dteday : date
# - season : season (1:springer, 2:summer, 3:fall, 4:winter)
# - yr : year (0: 2011, 1:2012)
# - mnth : month ( 1 to 12)
# - hr : hour (0 to 23)
# - holiday : weather day is holiday or not (extracted from [Web Link])
# - weekday : day of the week
# - workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
# + weathersit : 
# - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# - temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
# - atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
# - hum: Normalized humidity. The values are divided to 100 (max)
# - windspeed: Normalized wind speed. The values are divided to 67 (max)
# - casual: count of casual users
# - registered: count of registered users
# - cnt: count of total rental bikes including both casual and registered

# In[ ]:

# assume we want 2 features, 'temp' and 'hum' and weathersit, because
# weathersit is categorical we must make dummy variables
# create a new dataframe with only features temp and hum, 
# and all dummy variables required for weathersit (hint n - 1 classes)
# hint (pd.get_dummies(df_column)) will transform your variable 
# into n dummy classes
# hint, use .join to merge two dataframes on a common key (default inner join on index)
# call your features dataframe 'modeldata' and your y response column 'y'


# In[ ]:

# kf = cross_validation.KFold(len(modeldata), n_folds=5, shuffle=True)

# you must define modeldata in cell above for kf assignment to run
# kf returns a dictionary composed of key-value pairs


# In[ ]:

lm_obj = linear_model.LinearRegression() # instaniate only one time
scores = [] # we will append mse scores from each iteration in kf
for train_index, test_index in kf: # for (key,value) in dictionary
    x_train = modeldata.iloc[train_index] # get new set each iteration
    y_train = y.iloc[train_index]
    
    x_test = modeldata.iloc[test_index] # get new test data each iteration
    y_test = y.iloc[test_index]
        
    lm = lm_obj.fit(x_train, y_train) # fit new model each iteration
    x_test_pred = lm.predict(x_test)

    mse = metrics.mean_squared_error(y_test,x_test_pred) 
    # get new mse each iteration
    scores.append(mse) # append mse scores from each model to scores list


# In[ ]:

# print the mean mse score from all iterations, explain output


# In[ ]:

# fit a regression model on all the model and outcome data (modeldata and y)
# hint use, linear_model.LinearRegression().fit()
# get predictions from fitted model using same model data
# calculate MSE and interpret 
# hint MSE = SSE/n = (y - y_est)/n, y_est = lm.predict(modeldata)


# In[ ]:

# fit a lasso regression model on all the model and outcome data (modeldata and y)
# hint use, linear_model.Lasso().fit()
# get predictions from fitted model using same model data
# calculate MSE and interpret
# hint MSE = SSE/n = (y - y_est)/n, y_est = lm.predict(modeldata)


# In[ ]:

# fit a Ridge regression model on all the model and outcome data (modeldata and y)
# hint use, linear_model.Ridge().fit()
# get predictions from fitted model using same model data
# calculate MSE and interpret
# hint MSE = SSE/n = (y - y_est)/n, y_est = lm.predict(modeldata)

