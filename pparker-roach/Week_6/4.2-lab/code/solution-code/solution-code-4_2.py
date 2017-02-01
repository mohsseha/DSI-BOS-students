
# coding: utf-8

# # Natural Language Processing Lab
# 
# In this lab we will further explore Scikit's and NLTK's capabilities to process text. We will use the 20 Newsgroup dataset, which is provided by Scikit-Learn.

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[ ]:

from sklearn.datasets import fetch_20newsgroups


# In[ ]:


categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=('headers', 'footers', 'quotes'))

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=('headers', 'footers', 'quotes'))


# ## 1. Data inspection
# 
# We have downloaded a few newsgroup categories and removed headers, footers and quotes.
# 
# Let's inspect them.
# 
# 1. What data taype is `data_train`
# > sklearn.datasets.base.Bunch
# - Is it like a list? Or like a Dictionary? or what?
# > Dict
# - How many data points does it contain?
# - Inspect the first data point, what does it look like?
# > A blurb of text

# In[ ]:

type(data_train)


# In[ ]:

data_train.keys()


# In[ ]:

len(data_train['data'])


# In[ ]:

len(data_train['target'])


# In[ ]:

data_train['data'][0]


# ## 2. Bag of Words model
# 
# Let's train a model using a simple count vectorizer
# 
# 1. Initialize a standard CountVectorizer and fit the training data
# - how big is the feature dictionary
# - repeat eliminating english stop words
# - is the dictionary smaller?
# - transform the training data using the trained vectorizer
# - what are the 20 words that are most common in the whole corpus?
# - what are the 20 most common words in each of the 4 classes?
# - evaluate the performance of a Lotistic Regression on the features extracted by the CountVectorizer
#     - you will have to transform the test_set too. Be carefule to use the trained vectorizer, without re-fitting it
# - try the following 3 modification:
#     - restrict the max_features
#     - change max_df and min_df
#     - use a fixed vocabulary of size 80 combining the 20 most common words per group found earlier
# - for each of the above print a confusion matrix and investigate what gets mixed
# > Anwer: not surprisingly if we reduce the feature space we lose accuracy
# - print out the number of features for each model

# In[ ]:

from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:

cvec = CountVectorizer()
cvec.fit(data_train['data'])


# In[ ]:

len(cvec.get_feature_names())


# In[ ]:

cvec = CountVectorizer(stop_words='english')
cvec.fit(data_train['data'])
len(cvec.get_feature_names())


# In[ ]:

X_train = pd.DataFrame(cvec.transform(data_train['data']).todense(),
                       columns=cvec.get_feature_names())


# In[ ]:




# In[ ]:

names = data_train['target_names']
names


# In[ ]:

y_train = data_train['target']


# In[ ]:

common_words = []
for i in xrange(4):
    word_count = X_train[y_train==i].sum(axis=0)
    print names[i], "most common words"
    cw = word_count.sort_values(ascending = False).head(20)
#     cw.to_csv('../../../5.2-lesson/assets/datasets/'+names[i]+'_most_common_words.csv')
    print cw
    common_words.extend(cw.index)
    print 


# In[ ]:

X_test = pd.DataFrame(cvec.transform(data_test['data']).todense(),
                      columns=cvec.get_feature_names())


# In[ ]:

y_test = data_test['target']


# In[ ]:

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)


# In[ ]:

from sklearn.metrics import confusion_matrix, accuracy_score


# In[ ]:

def docm(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    if labels is not None:
        cols = ['p_'+c for c in labels]
        df = pd.DataFrame(cm, index=labels, columns=cols)
    else:
        cols = ['p_'+str(i) for i in xrange(len(cm))]
        df = pd.DataFrame(cm, columns=cols)
    return df


# In[ ]:

from sklearn.pipeline import make_pipeline

model = make_pipeline(CountVectorizer(stop_words='english',
                                      max_features=1000),
                      LogisticRegression(),
                      )
model.fit(data_train['data'], y_train)
y_pred = model.predict(data_test['data'])
print accuracy_score(y_test, y_pred)
docm(y_test, y_pred, names)
print "Number of features:", len(model.steps[0][1].get_feature_names())


# In[ ]:

model = make_pipeline(CountVectorizer(stop_words='english',
                                      max_features=1000,
                                      min_df=0.03),
                      LogisticRegression(),
                      )
model.fit(data_train['data'], y_train)
y_pred = model.predict(data_test['data'])
print accuracy_score(y_test, y_pred)
docm(y_test, y_pred, names)
print "Number of features:", len(model.steps[0][1].get_feature_names())


# In[ ]:

model = make_pipeline(CountVectorizer(stop_words='english',
                                      vocabulary=set(common_words)),
                      LogisticRegression(),
                      )
model.fit(data_train['data'], y_train)
y_pred = model.predict(data_test['data'])
print accuracy_score(y_test, y_pred)
docm(y_test, y_pred, names)
print "Number of features:", len(model.steps[0][1].get_feature_names())


# ## 3. Hashing and TF-IDF
# 
# Let's see if Hashing or TF-IDF improves the accuracy.
# 
# 1. Initialize a HashingVectorizer and repeat the test with no restriction on the number of features
# - does the score improve with respect to the count vectorizer?
#     - can you change any of the default parameters to improve it?
# - print out the number of features for this model
# - Initialize a TF-IDF Vectorizer and repeat the analysis above
# - can you improve on your best score above?
#     - can you change any of the default parameters to improve it?
# - print out the number of features for this model

# In[ ]:

from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer


# In[ ]:

model = make_pipeline(HashingVectorizer(stop_words='english',
                                        non_negative=True,
                                        n_features=2**16),
                      LogisticRegression(),
                      )
model.fit(data_train['data'], y_train)
y_pred = model.predict(data_test['data'])
print accuracy_score(y_test, y_pred)
docm(y_test, y_pred, names)
print "Number of features:", 2**16


# In[ ]:

model = make_pipeline(TfidfVectorizer(stop_words='english',
                                      sublinear_tf=True,
                                      max_df=0.5,
                                      max_features=1000),
                      LogisticRegression(),
                      )
model.fit(data_train['data'], y_train)
y_pred = model.predict(data_test['data'])
print accuracy_score(y_test, y_pred)
docm(y_test, y_pred, names)
print "Number of features:", len(model.steps[0][1].get_feature_names())


# ## 4. Classifier comparison
# 
# Of all the vectorizers tested above, choose one that has a reasonable performance with a manageable number of features and compare the performance of these models:
# 
# - KNN
# - Logistic Regression
# - Decision Trees
# - Support Vector Machine
# - Random Forest
# - Extra Trees
# 
# In order to speed up the calculation it's better to vectorize the data only once and then compare the models.

# In[ ]:

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


# In[ ]:

models = [KNeighborsClassifier(),
          LogisticRegression(),
          DecisionTreeClassifier(),
          SVC(),
          RandomForestClassifier(),
          ExtraTreesClassifier()]

tvec = TfidfVectorizer(stop_words='english',
                       sublinear_tf=True,
                       max_df=0.5,
                       max_features=1000)

tvec.fit(data_train['data'])
X_train = tvec.transform(data_train['data'])
X_test = tvec.transform(data_test['data'])


# In[ ]:

res = []

for model in models:
    print model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print score
    cm = docm(y_test, y_pred, names)
    print cm
    res.append([model, score])

# pd.DataFrame(res, columns=['model', 'score']).to_csv('../../../5.2-lesson/assets/datasets/20newsgroups/model_comparison.csv')


# ## Bonus: Other classifiers
# 
# Adapt the code from [this example](http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#example-text-document-classification-20newsgroups-py) to compare across all the classifiers suggested and to display the final plot

# In[ ]:

from time import time

from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics

feature_names = np.array(tvec.get_feature_names())

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

###############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if feature_names is not None:
            print("top 10 keywords per class:")
            for i, category in enumerate(categories):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s"
                      % (category, " ".join(feature_names[top10]))))
        print()


    print("classification report:")
    print(metrics.classification_report(y_test, pred,
                                        target_names=categories))


    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
  ('classification', LinearSVC())
])))

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='r')
plt.barh(indices + .3, training_time, .2, label="training time", color='g')
plt.barh(indices + .6, test_time, .2, label="test time", color='b')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()


# ## Bonus: NLTK
# 
# NLTK is a vast library. Can you find some interesting bits to share with classmates?
# Start here: http://www.nltk.org/
