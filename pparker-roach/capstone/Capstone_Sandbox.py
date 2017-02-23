
# coding: utf-8

# <center><h1>General Assembly Data Science Immersion Program</h1>
# 
# <h1>Capstone Project</h1>
# <h1>Is This Mushroom Edible or Poisonous</h1></center>

# # Executive Summary 

# ## Problem Statement
# <>
# 
# 
# ## Goal
# <> 
#    
# ## Deliverables
#     
#     
# ## Approach
# 
# <img src="model_selection_flowchart_stackoverflow.png">

# # Summary of Findings

# tbd

# ## Supporting Graphics

# tbd

# # Recommendations for Next Steps

# As with many exploritory projects, new approaches and potential applications come to mind. In this case, the task is to find the salient features needed to accurately, if possible, identify a mushroom as poisonous or not. A expansion of the scope of such an effort could be to identify the mushroom itself, with the side effect of knowing it's edibility. I believe that a rule-based expert systems approach may be better suited to the task. One could load up an expert system's working memory with known mushrooms and their attributes along with whatever is known regarding a new sample. The rules could be structured to find the best match given the known sample data and ask directed questions to the user to refine its prediction. In this way, new known mushrooms and their characteristics could be added at will to expand the range of identifiable mushrooms without having to change the underlying rule logic. The results of this current study could feed the selection of the critical features and characteristics to hone in on.

# ## Data Description
# 
# Sources: 
#     (a) Mushroom records drawn from The Audubon Society Field Guide to North
#         American Mushrooms (1981). G. H. Lincoff (Pres.), New York: Alfred
#         A. Knopf
#     (b) Donor: Jeff Schlimmer (Jeffrey.Schlimmer@a.gp.cs.cmu.edu)
#     (c) Date: 27 April 1987
# 
# Relevant Information:
#     This data set includes descriptions of hypothetical samples
#     corresponding to 23 species of gilled mushrooms in the Agaricus and
#     Lepiota Family (pp. 500-525).  Each species is identified as
#     definitely edible, definitely poisonous, or of unknown edibility and
#     not recommended.  This latter class was combined with the poisonous
#     one.  The Guide clearly states that there is no simple rule for
#     determining the edibility of a mushroom; no rule like ``leaflets
#     three, let it be'' for Poisonous Oak and Ivy.
# 
# Number of Instances: 8124
# 
# Number of Attributes: 22 (all nominally valued)
# 
# Logical rules for the mushroom data sets.
# 
# 	Logical rules given below seem to be the simplest possible for the
# 	mushroom dataset and therefore should be treated as benchmark results.
# 
# 	Disjunctive rules for poisonous mushrooms, from most general
# 	to most specific:
# 
# 	P_1) odor=NOT(almond.OR.anise.OR.none)
# 	     120 poisonous cases missed, 98.52% accu
# 
# 	P_2) spore-print-color=green
# 	     48 cases missed, 99.41% accuracy
#          
# 	P_3) odor=none.AND.stalk-surface-below-ring=scaly.AND.
# 	          (stalk-color-above-ring=NOT.brown) 
# 	     8 cases missed, 99.90% accuracy
#          
# 	P_4) habitat=leaves.AND.cap-color=white
# 	         100% accuracy     
# 
# 	Rule P_4) may also be
# 
# 	P_4') population=clustered.AND.cap_color=white
# 
# 	These rule involve 6 attributes (out of 22). Rules for edible
# 	mushrooms are obtained as negation of the rules given above, for
# 	example the rule:
# 
# 	odor=(almond.OR.anise.OR.none).AND.spore-print-color=NOT.green
# 	gives 48 errors, or 99.41% accuracy on the whole dataset.
# 
# 	Several slightly more complex variations on these rules exist,
# 	involving other attributes, such as gill_size, gill_spacing,
# 	stalk_surface_above_ring, but the rules given above are the simplest
# 	we have found.
# 
# 
# Attribute Information: (classes: edible=e, poisonous=p)
#      1. cap-shape:                bell=b,conical=c,convex=x,flat=f,
#                                   knobbed=k,sunken=s
#      2. cap-surface:              fibrous=f,grooves=g,scaly=y,smooth=s
#      3. cap-color:                brown=n,buff=b,cinnamon=c,gray=g,green=r,
#                                   pink=p,purple=u,red=e,white=w,yellow=y
#      4. bruises?:                 bruises=t,no=f
#      5. odor:                     almond=a,anise=l,creosote=c,fishy=y,foul=f,
#                                   musty=m,none=n,pungent=p,spicy=s
#      6. gill-attachment:          attached=a,descending=d,free=f,notched=n
#      7. gill-spacing:             close=c,crowded=w,distant=d
#      8. gill-size:                broad=b,narrow=n
#      9. gill-color:               black=k,brown=n,buff=b,chocolate=h,gray=g,
#                                   green=r,orange=o,pink=p,purple=u,red=e,
#                                   white=w,yellow=y
#     10. stalk-shape:              enlarging=e,tapering=t
#     11. stalk-root:               bulbous=b,club=c,cup=u,equal=e,
#                                   rhizomorphs=z,rooted=r,missing=?
#     12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
#     13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
#     14. stalk-color-above-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
#                                   pink=p,red=e,white=w,yellow=y
#     15. stalk-color-below-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
#                                   pink=p,red=e,white=w,yellow=y
#     16. veil-type:                partial=p,universal=u
#     17. veil-color:               brown=n,orange=o,white=w,yellow=y
#     18. ring-number:              none=n,one=o,two=t
#     19. ring-type:                cobwebby=c,evanescent=e,flaring=f,large=l,
#                                   none=n,pendant=p,sheathing=s,zone=z
#     20. spore-print-color:        black=k,brown=n,buff=b,chocolate=h,green=r,
#                                   orange=o,purple=u,white=w,yellow=y
#     21. population:               abundant=a,clustered=c,numerous=n,
#                                   scattered=s,several=v,solitary=y
#     22. habitat:                  grasses=g,leaves=l,meadows=m,paths=p,
#                                   urban=u,waste=w,woods=d
# 

# ## The Model

# tbd

# Datasets used for various models
# 
#     * original            = clean copy of the original data for chain of custody purposes
#     
#     * shrooms_rules_X     = 80% of training data for reduced dimensionality based on rule data
#     * shrooms_rules_y     = 80% of target data for reduced dimensionality based on rule data
#     * shrooms_rules_test  = 80% of testing data for reduced dimensionality based on rule data
# 
#     * shrooms_scv_X       = 80% of training data
#     * shrooms_scv_y       = 80% of target data
#     * shrooms_scv_test    = 20 % of testing data
# 
#     * shrooms_knn_X        80% of training data
#     * shrooms_knn_y       = 80% of target data
#     * shrooms_knn_test    = 20 % of testing data
# 
#     * shrooms_scve_X      = 80% of training data
#     * shrooms_scve_y      = 80% of target data
#     * shrooms_scve_test   = 20 % of testing data
# 
# 
# Models
#   * SCV Linear
#     * shrooms_scv_X    = 80% of training data
#     * shrooms_scv_y    = 80% of target data
#     * shrooms_scv_test = 20 % of testing data
#   * kNN
#     * shrooms_knn_X    = 80% of training data
#     * shrooms_knn_y    = 80% of target data
#     * shrooms_knn_test = 20 % of testing data
#   * SVC Ensemble (maybe)
#     * shrooms_scve_X    = 80% of training data
#     * shrooms_scve_y    = 80% of target data
#     * shrooms_scve_test = 20 % of testing data
# 
# Feature Reduction (combination of)
#   * rule information given with data
#   * personal a priori knowledge
#   * PCA
#   
# For each model
#   * standardization/normalization
#   * cross validation
#   * grid search
#   * test on reserved 20% of training data
#   * evaluate on accuracy and confusion matrix

# #### The following block is used to import all Python libraries used in this model

# In[2]:

import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns


from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA

get_ipython().magic('matplotlib inline')


# #### The following block imports the mushroom classification data stored in mushrooms.csv and stores it in a dataframe called <i><b>original</i></b>. Each model will start with a copy of original and manipulate the data to suit the needs of the model. The original data will be preserved in the spirit of "chain of custody of evidence"

# In[3]:

original = pd.read_csv("mushrooms.csv") #original will be kept for historical purposes


# We willl use a dataframe called shrooms to do some initial exploration of our data

# In[4]:

shrooms = original.copy()
# from sklearn.model_selection import train_test_split

# train, test = train_test_split(df, test_size = 0.2)


# In[ ]:




# #### Let's take a look at some of the features of this data

# In[5]:

shrooms.head()


# In[6]:

shrooms.shape


# In[7]:

shrooms.columns


# In[8]:

corrmat = shrooms.corr()
f, ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmat, vmax = .8, square=True)
# networks = corrmat.columns.get_level_values("network")
# for i, network in enumerate(neworks):
#     if i and network != networks[i - 1]:
#         ax.axhline(len(networks) - i, c="w")
#         ax.axvline(i, c="w")
f.tight_layout()


# In[ ]:

shrooms.describe()


# #### The data is encoded as an alpha character which maps to an attribute as described in the Data Description section above. Let's convert those alpha characters to dummy columns in order to start looking for correlations.
# 

# In[ ]:

shrooms.columns


# In[10]:

columns = shrooms.columns
columns = columns.drop('class') # we won't make dummy columns for our target column
for col in columns:
    for data in shrooms[col].unique():
        shrooms[col + "_" + data] = shrooms[col] == data

# we can now drop the original columns
for col in columns:
    shrooms.drop(col, inplace = True, axis = 1)



# In[11]:

shrooms.head(20)


# In[10]:

# and now let's declare that in our target variable 'class' that p=1 (poisonous) and e=0 (edible)
# and convert the data in y accordingly

shrooms['class'] = shrooms['class'] == 'p'


# In[ ]:




# In[11]:

shrooms.head()


# In[12]:

# Now let's replace all of the bool values (True/False) with integers (1/0)
# columns = shrooms.columns
# for col in columns:
#     print(col)
#     shrooms[col] = shrooms[col].astype(int)
shrooms = shrooms.astype(int)


# In[13]:

shrooms.head()


# #### Let's do a heatmap of a correlation matrix of all of these characteristics to see if there is anything striking.

# In[14]:

corrmat = shrooms.corr()


# In[15]:

f, ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmat, vmax = .8, square=True)
# networks = corrmat.columns.get_level_values("network")
# for i, network in enumerate(neworks):
#     if i and network != networks[i - 1]:
#         ax.axhline(len(networks) - i, c="w")
#         ax.axvline(i, c="w")
f.tight_layout()


# In[16]:

# Generate a mask for the upper triangle
mask = np.zeros_like(corrmat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corrmat, mask=mask, cmap=cmap, vmax=.3,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


# #### Let's see if we can find anything interesting about the attributes using a Support Vector Machine

# In[17]:

#first let's separate the training (X) from the target (y) data
X = shrooms.drop('class', axis=1)
y = shrooms['class']


# In[18]:

model = SVC(kernel='linear')
model.fit(X, y)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)


# #### Cross Validation

# In[19]:

for i in range(2, 8):
    cvscores = cross_val_score(model, X, y, cv = i, n_jobs=-1)
    print ("CV score: ", cvscores.mean(), " +/- ", cvscores.std(), " cv = ", i)


# #### Grid Search

# In[20]:

from sklearn.grid_search import GridSearchCV
parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 3, 10]}
clf = GridSearchCV(model, parameters, n_jobs=-1)
clf.fit(X, y)
clf.best_estimator_


# #### KNN

# In[21]:

knn = KNeighborsClassifier(n_neighbors=5,
                           weights='uniform',
                           p=2,
                           metric='minkowski')


# In[22]:

accuracies = cross_val_score(knn, X, y, cv=5)

print("K-Fold accuracies:", accuracies)
print("Mean accuracy:", accuracies.mean())


# #### let's look at a pair plot using the original data

# In[23]:

shrooms_cat = original.copy()


# In[24]:

columns_cat = shrooms_cat.columns


# In[25]:

for col in columns_cat:
    shrooms_cat[col] = shrooms_cat[col].astype("category")
    shrooms_cat[col] = shrooms_cat[col].cat.codes
shrooms_cat.head()


# In[26]:

X = shrooms_cat.drop('class', axis = 1)
y = shrooms_cat['class']


# #### histogram

# In[31]:

for col in shrooms_cat.columns:
    shrooms_cat[col].hist()
    plt.show()


# In[169]:

shrooms_cat.columns


# In[172]:

# Let me put in some of my own a priori knowledge of important feathres 
g = sns.PairGrid(shrooms_cat[['class','bruises','gill-attachment','gill-color', 'veil-type', 'spore-print-color', 'habitat']])
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=7); 


# In[36]:

'''There   are   many   ways   of   measuring   classification   performance.Accuracy, confusion matrix, log-loss, 
and AUC are some of the mostpopular metrics. Precision-recall is also widely used; I’ll explain it in
“Ranking Metrics” on page 12.7

sklearn.metrics.confusion_matrix¶
Examples
>>>
from sklearn.metrics import confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
confusion_matrix(y_true, y_pred)



>>>
y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])

sklearn.metrics.accuracy_score¶

Example
import numpy as np
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
accuracy_score(y_true, y_pred)

accuracy_score(y_true, y_pred, normalize=False)

sklearn.metrics.auc

Example
import numpy as np
from sklearn import metrics
y = np.array([1, 1, 2, 2])
pred = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
metrics.auc(fpr, tpr)

look at PCA

set aside 20% of data for testing
'''


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

print(X.shape)
X_new = SelectKBest(chi2, k=8).fit_transform(X, y)
X_new.shape
foo = pd.DataFrame(X_new)


# In[38]:

foo.columns



# In[39]:

from sklearn.decomposition import PCA



pca = PCA(n_components=5)
pca.fit(X)


# In[40]:

pca


# In[42]:

pca.components_


# In[ ]:

import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
from sklearn.svm import SVC
clf = SVC()
clf.fit(X, y) 




print(clf.predict([[-0.8, -1]]))


# In[4]:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

# datasets = [make_moons(noise=0.3, random_state=0),
#             make_circles(noise=0.2, factor=0.5, random_state=1),
#             linearly_separable
#             ]
datasets = [(shrooms_X, shrooms_y),               # all characteristics
           (shrooms_X[["odor",                    # only important characteristics for rule-based classification
                      "spore-print-color", 
                      "stalk-surface-above-ring", 
                      "stalk-color-above-ring", 
                      "habitat", 
                      "population", 
                      "cap-color" ]], shrooms_y),
           (shrooms_X[['odor',
                       'spore-print-color',
                       'stalk-surface-above-ring',
                       'habitat',
                       'population',
                       'cap-color',
                       'cap-shape', 
                       'cap-surface', 
                       'gill-size',
                       'gill-spacing' ]], shrooms_y)
            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test =         train_test_split(X, y, test_size=.4, random_state=42)

    
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    print(x_min,x_max)
    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()


# In[ ]:

import matplotlib.pyplot as plt
import pandas as pd
import itertools
#data
df = pd.DataFrame(
    {'id': [1, 2, 3, 3],
     'labels': ['HPRR1234', 'HPRR4321', 'HPRR2345', 'HPRR2345'],
     'g': ['KRAS', 'KRAS', 'ELK4', 'ELK4'],
     'r1': [15, 9, 15, 1],
     'r2': [14, 8, 7, 0],
     'r3': [14, 16, 9, 12]})
#extra setup
plt.rcParams['xtick.major.pad'] = 8
#plotting style(s)
marker = itertools.cycle((',', '+', '.', 'o', '*'))
color = itertools.cycle(('b', 'g', 'r', 'c', 'm', 'y', 'k'))
#plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(df['id'], df['r1'], ls='', ms=10, mew=2,
        marker=marker.next(), color=color.next())
ax.plot(df['id'], df['r2'], ls='', ms=10, mew=2,
        marker=marker.next(), color=color.next())
ax.plot(df['id'], df['r3'], ls='', ms=10, mew=2,
        marker=marker.next(), color=color.next())
# set the tick labels
ax.xaxis.set_ticks(df['id'])
ax.xaxis.set_ticklabels(df['labels'])
plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=12)
plt.tight_layout()


# In[13]:

primary_cats = ["odor", "spore-print-color", "stalk-surface-above-ring", "stalk-color-above-ring", "habitat", "population", "cap-color"]
for cat in primary_cats:
    # Create a Pairplot
    g = sns.pairplot(shrooms_plot,hue="class",palette="muted",size=5, 
        vars=['class', cat ],kind='reg')

    # To change the size of the scatterpoints in graph
    g = g.map_offdiag(plt.scatter,  s=35,alpha=0.5)

    # remove the top and right line in graph
    sns.despine()
    # Additional line to adjust some appearance issue
    plt.subplots_adjust(top=0.9)

    # Set the Title of the graph from here
    g.fig.suptitle('Relation between Class and ' + cat, 
        fontsize=34,color="b",alpha=0.3)


# In[7]:

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

actual = [1,1,1,0,0,0]
predictions = [0.9,0.9,0.9,0.1,0.1,0.1]

false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# 

# In[ ]:



