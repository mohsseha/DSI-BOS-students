
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import patsy

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.grid_search import GridSearchCV


# In[7]:

# prep data, convert date to datetime, split for train and test, and build model
sf_crime = pd.read_csv('C:/Users/Pat.NOAGALLERY/Documents/data_sources/sf_crime_train.csv')

sf_crime = sf_crime.dropna()

sf_crime['Dates'] = pd.to_datetime(sf_crime.Dates)
sf_crime_dates = pd.DatetimeIndex(sf_crime.Dates.values, dtype='datetime64[ns]', freq=None)

sf_crime['hour'] = sf_crime_dates.hour
sf_crime['month'] = sf_crime_dates.month
sf_crime['year'] = sf_crime_dates.year



# In[20]:

subset = ['VEHICLE THEFT','BURGLARY','DRUG/NARCOTIC']
sf_crime_sub = sf_crime[sf_crime['Category'].str.contains('|'.join(subset))]

#sf_sample = sf_crime_sub.sample(frac=0.50)

X = patsy.dmatrix('~ C(hour) + C(DayOfWeek) + C(PdDistrict)', sf_crime_sub)
Y = sf_crime_sub.Category.values

# split for train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, stratify=Y, random_state=77)


# In[34]:

Test=patsy.dmatrix('~ C(DayOfWeek)',sf_crime_sub)
Test[70]


# In[46]:

# fit model with five folds and lasso regularization
# use Cs=15 to test a grid of 15 distinct parameters
# remeber: Cs describes the inverse of regularization strength
logreg_cv = LogisticRegressionCV(solver='liblinear', penalty="l1", Cs=15, cv=5, scoring='accuracy') # update inputs here
logreg_cv.fit(X_train, Y_train)


# In[43]:

# find best C per class
print('best C for class:')
best_C = {logreg_cv.classes_[i]:x for i, (x, c) in enumerate(zip(logreg_cv.C_, logreg_cv.classes_))}
print(best_C)


# In[47]:

# fit regular logit model to 'DRUG/NARCOTIC' and 'BURGLARY' classes
# use lasso penalty
logreg_1 = LogisticRegression(C=best_C['DRUG/NARCOTIC'], penalty='l1', solver='liblinear')
logreg_2 = LogisticRegression(C=best_C['BURGLARY'], penalty='l1', solver='liblinear')

logreg_1.fit(X_train, Y_train)
logreg_2.fit(X_train, Y_train)


# In[48]:

# build confusion matrices for the models above
Y_1_pred = logreg_1.predict(X_train)
Y_2_pred = logreg_2.predict(X_train)



conmat_1 = confusion_matrix(Y_train, Y_1_pred, labels=logreg_1.classes_)
conmat_1 = pd.DataFrame(conmat_1, columns=logreg_1.classes_, index=logreg_1.classes_)

conmat_2 = confusion_matrix(Y_train, Y_2_pred, labels=logreg_2.classes_)
conmat_2 = pd.DataFrame(conmat_2, columns=logreg_2.classes_, index=logreg_2.classes_)


# In[50]:

# print classification reports
print(conmat_1)
print(conmat_2)


# In[59]:

# run gridsearch using GridSearchCV and 5 folds
# score on f1_macro; what does this metric tell us?
logreg = LogisticRegression()
C_vals = [0.0001, 0.001, 0.01, 0.1, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0, 100.0, 1000.0]
penalties = ['l1','l2']

gs = GridSearchCV(logreg, {'penalty':penalties, 'C':C_vals}, verbose=True, cv=5, scoring='f1_macro')
gs.fit(X,Y)


# In[61]:

# find the best parameter
gs.best_params_


# In[69]:

# use this parameter to .fit, .predict, and print a classification_report for our X and Y
gs.fit(X_train, Y_train)
gs_pred = gs.predict(X_train)
conmat_gs = confusion_matrix(Y_train, gs_pred, labels=gs.classes_)
#conmat_gs = pd.DataFrame(gs, columns=gs.classes_)

#conmat_1 = confusion_matrix(Y_train, Y_1_pred, labels=logreg_1.classes_)
#conmat_1 = pd.DataFrame(conmat_1, columns=logreg_1.classes_, index=logreg_1.classes_)


# In[ ]:



