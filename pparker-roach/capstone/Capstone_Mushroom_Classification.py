
# coding: utf-8

# <center><h1>General Assembly Data Science Immersion Program</h1>
# 
# <h1>Capstone Project</h1>
# <h1>Safe to Eat or Deadly Poisonous?</h1></center>

# # Executive Summary 
# 
# General Assembly is a pioneer in education and career transformation, specializing in today’s most in-demand skills. They are a leading source for training, staffing, and career transitions, fostering a flourishing community of professionals pursuing careers they love.
# 
# One of General Assembly's offerings is a Data Science Immersions program. It is an intensive 12-week program which focuses on hands-on experience in the various people, business and technical aspects of the field of Data Science. Participants in the program are required to produce a Capstone Project which highlights the Data Science skills they acquired.
# 
# This technical report documents my Capstone Project.
# 
# ## Problem Statement
# 
# This problem is taken from the Kaggle Data Science Competition website. In 2010, Kaggle was founded as a platform for predictive modelling and analytics competitions on which companies and researchers post their data and statisticians and data miners from all over the world compete to produce the best models.
# 
# I selected the "Mushroom Classification - Save to Eat of Deadly Poisonous" competition for a couple of reasons. First, the competition supplied a very comprehensive dataset for analysis. Secondly, I have been an avid amateur mycologist for 30 years with a primary focus of putting wild mushrooms on my table. This project offered me an opportunity to bring my own a priori knowledge to the problem while, at the same time, providing a mechanism for me to broaden my knowledge and understanding of the discipline. My motivation for this is purely from a survival perspective as illustrated by the following quote...
# 
# <b><i>There are old mushroom hunters and bold mushroom hunters, but very few old, bold mushroom hunters!</i></b>
# 
# There are tales of whole families found dead around thier dinner tables with mushrooms as part of their last meal!
# 
# The following Context and Content sections are taken directly from the Kaggle website.
# 
# ### Context
# 
# Although this dataset was originally contributed to the UCI Machine Learning repository nearly 30 years ago, mushroom hunting (otherwise known as "shrooming") is enjoying new peaks in popularity. Learn which features spell certain death and which are most palatable in this dataset of mushroom characteristics. And how certain can your model be?
# 
# ### Content
# 
# This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like "leaflets three, let it be'' for Poisonous Oak and Ivy.
# 
# ## Goal
# 
# The initial goal of this project was to develop a model that provided a very high probability of classifying mushroom as with Edible or Poisonous given the supplied dataset. Given information supplied with the dataset and my own a priori knowledge of the subject, my initial modeling proved very successful. Given that I chose to run the models over a number of classifying modeling techniques to ascertain which worked best given the nature of the supplied data. The benefits are two-fold; it afforded me the ability to pick the best model for my analysis and it helped exercise and solidify my understanding of the various modeling techniques.
#    
#    

# # Identification of outliers
# 
# This is a classification problem with a well defined dataset that is entirely categorical in nature. As such, it inherently does not suffer from outliers. However, a close cousin to outliers in classification problems is <b><i>imbalanced classes</i></b>. Imbalanced classes in this context means that the training classifiers are skewed in one direction or another. An example in a binary classification system such as is the case with this mushroom classification problem would be that the training set would have 90% of the mushroom classed as, let's say, poisonous. This would severly skew our confidence in the model.
# 
# The dataset presented for this case contains 8124 samples; 3916 of which are identified as poisonous. This represents 48%-52% split between poisonous and edible samples raising no imbalanced classes issues.
# 

# # Data Description
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
#     determining the edibility of a mushroom; no rule like "leaflets
#     three, let it be" for Poisonous Oak and Ivy.
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
# ## How the data was used in this project
# 
# 
# Each of the models selected for testing were fed four different datasets derived from the original. Each was into training (80%) and test (20%) sets. Here are is a desciption of the four sets and the logic behind them.
# 
# * shrooms
#     The shrooms datasets are an exact duplicate of the original dataset which are then processed. The columns (attributes) remained the same but the data was converted to categorical integers; i.e., in the description above, the ring-number's values were converted from n, o and t to 0, 1 and 2. Two version of this shrooms dataset were fed to the models:
#     * shrooms
#         This is the dataset as described above with all of the attributes. This is a a baseline to start with.
#     * shrooms with reduced attributes
#         
#         In the data description above you will notice a section titled: Logical rules for the mushroom data sets. The rules detail 4 or 5 rules that look at specific occurances of attribute/value pairs that are very good predictors of whether a mushroom is poisonous or edible. A collection of those attribures were selected from the original dataset and the rest were dropped. Those attribures are...
#             *  "odor"
#             *  "spore-print-color"
#             *  "stalk-surface-above-ring"
#             *  "stalk-color-above-ring"
#             *  "habitat"
#             *  "population"
#             *  "cap-color" 
#     * shrooms_dummy
#         
#         To further refine our findings, the shrooms data was converted into a 'dummy categories' dataset; that is, each attribute/value combination was given a unique column in our dataset and a value of 1 or 0 (present or not). Given this information we can see if model performance was affected by attribute/value combinations or just by the attributes themselves.
#         
#         
#     *  shrooms_dummy with reduced attribute/value combinations
#     
#         Given the 'rules' information above, only attribute/value combinations contained in the rules are included in this version of the data that is supplied to the models. These attribute/value combinations are as follows:
#         *  'cap-color_w'
#         *  'odor_a'
#         *  'odor_l'
#         *  'odor_n'
#         *  'stalk-surface-below-ring_y'
#         *  'stalk-color-above-ring_n'
#         *  'spore-print-color_r'
#         *  'population_c'
#         *  'habitat_l'
#         

# # Model Selection and Implementation
# 
# ## Initial Approach
# 
# While research appropriate models for use in this binary classification problem I stumbled upon the following graphic at StackOverflow.com...
# 
# <img src="model_selection_flowchart_stackoverflow.png" >
# 
# This led me to try 'Linear SVC' and 'K Neighbors' classifiers. Both of these models gave me very strong positive results, but while researching evaluation techniques I found a technique that rotated over a number of classifier models.
# 
# ## Final Approach
# 
# Given the technique I found for iterating over a number of classifier models, some of which I was not familiar with, I decided to apply each of the datasets I prepared (see above) to each of the models I found. Those models are...
# 
#     *  "Nearest Neighbors"
#     *  "Linear SVM"
#     *  "RBF SVM"
#     *  "Gaussian Process
#     *  "Decision Tree"
#     *  "Random Forest" 
#     *  "Neural Net"
#     *  "AdaBoost"
#     *  "Naive Bayes"
#     *  "QDA"
# 

# # Visualizations & Statistical Analysis
# 
# It turns out that running 4 datasets against 10 models is a very CPU/memory intensive task that need to run overnight to complete. I had not budgeted tghe time for this and the final visualizations are not complete. I have the data stored in a CSV file necessary to create the graphics and will update this section in the very near future with those visualizations. Here is an example of the data collected on each of the runs of the model. In this case, it is the Nearest Neighbors model ran against the full shrooms_dummy database
# 
# 
# <table style="width:100%">
#   <tr>
#     <th></th>
#     <th></th> 
# 
#   </tr>
#   <tr>
#     <td>classifier</td>
#     <td>Nearest Neighbors</td> 
# 
#   </tr>
# 
#   <tr>
#     <td>score</td>
#     <td>1.000000</td> 
# 
#   </tr><tr>
#     <td>confusion matrix</td>
#     <td>[[1678, 0], [0, 1572]]</td> 
# 
#   </tr>
#     <tr>
#     <td>roc_auc_score</td>
#     <td>1.000000</td> 
# 
#   </tr>
#     <tr>
#     <td>false_positive_rate</td>
#     <td>[0.0, 1.0]</td> 
# 
#   </tr>
#  
#     <tr>
#     <td>true_positive_rate</td>
#     <td>[1.0, 1.0]</td> 
# 
#   </tr>
# </table>
# 
# Some Examples of visualizations coming...
# 
# <img src="training_cross_val_score.png">
# 
# <img src="correlations.png">
# 
# <img src="more_correlations.png">

# # Interpretation of Findings & Relation to Goals/Success Metrics
# 
# Will complete this after the visualizations are produced.

# # Stakeholder Recommendations & Next Steps 
# 
#   ### Stakeholders & Whoever may use this model
#   
#   Any good schroomer will tell you to never trust one source of information when considering eating a mushroom that you collect in the wild. This model may get you into the ballpark, but it is no substiture for classifying a mushroom down to not only a Poisonous/Edible category, but down to the exact species you are looking at. Like anything else in nature, there are variations on what you might see; color, size, etc. I recommend the most current version of The Audubon Society Field Guide to North American Mushrooms for classifying your mushroom. There is an identificaiton section in the back of the book that has you first take a spore print, and then with the color in hand, identify a few features; habitat, veil type, geographical region, etc. Once you complete this process, you are directed to a textural description of the mushroom in question where you can further refine your classification. In this description it will tell you the edibility of a mushroom; poisonous, choice, good, bad, unknown. Just as important, it will tell you if there are any close look-alikes which you should also read up on. You are also then directed to vary beautifly presented photographs of the mushrooms and look-alikes. The photographs should be the last thing you consider.
#   
#   Once you think you have identified the mushroom and decided that it may find its way to your table, take the following precautions:
#   
#   *  Only eat a small portion of the mushroom on your first try. Even if it is not poisonous, some people have alergies to different mushrooms, so beware. If you have no adverse affects, try some more the next day and keep increasing your consumption until you are confident.
#   *  Whenever you prepare and eat mushrooms (never eat wild mushrooms raw), alway keep a few samples in the refrigerator in case you get sick. If you do get sick, bring the sample to the emergency room with you. Different poisonous mushroom affect the body in different ways (nervous system, gastral, etc.) and there is no one way of treating the poison. Hospital staff can analyse an actual mushroom much better than a description. I started shrooming while living in France where all pharmacists are trained in mushroom classificaiton.
#   *  Alcohol consumption is not recommended while eating wild mushrooms.
#   
# REMEMBER - <b><i>There are old mushroom hunters, and bold mushroom hunters, but very few old bold mushroom hunters!</i></b>
#   
# ### Next Step Recommendations
#   1. Geographical region that a mushroom is native to is a very powerful attribute for identifying a particular mushroom. It would be very interesting to have that data added to see if it would be helpful in identifying a mushroom's edibility.
#   2. The rules provided with the dataset provide a very powerful mechanism for classifying the edibility of a mushroom. I spent a decade of my earlier career as a Knowledge Engineer building rule-based expert systems. From what I have learned in the execution of this project I believe that an expert systems approach would be very helpful in identifying a specific mushroom, and hence its edibility. I believe that a small rule set could be constructed to make a very accurate identification (see cautions above) and specific mushroom data could be added to expert systems working memory (database) as one collected mushrooms in thier local. Just a thought to think!
#     

# # Source Code
# An attempt has been made to make the Python 3 code (below) self documenting. It needs some cleaning up at the time of this writing but will be improved upon after the visualization routines are complete.

# #### The following block is used to import all Python libraries used in this model

# In[1]:

import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import random


from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix,  accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

get_ipython().magic('matplotlib inline')


# #### First let's load in the csv formatted dataset into a pandas dataframe named - original
# This 'original' dataframe will not be modified throughout this project and preserved for future analysis if needed in the spirit of "chain of custody". The mushrooms.csv dataset was obtained from the Kaggle.com machine learning competition website.

# In[2]:

original = pd.read_csv("mushrooms.csv") #original will be kept for historical purposes


# #### Next the 'original' dataset will be split into the data and the target components
#   * original_X        = training data representing 80% of the original
#   * original_y        = the target data

# In[3]:

original_X = original.drop('class', axis=1)
original_y = original['class']

original_X.shape, original_y.shape #ensure that the data and test are the same length


# #### Time to take a look at the data

# In[5]:

original_X.head()


# In[6]:

original_y.head()


# Given that the target data is supposed to contain either a p (poisonous) or e (edible) value, let's test to make sure that this is the case.

# In[7]:

class_is_good = True
for val in original_y:
    if val != 'p' and val != 'e':
        class_is_good = False
class_is_good


# #### Now we can start manipulating the data 

# First let's see if there are any missing data in the data dataframe

# In[8]:

original_X.isnull().sum().sum()


# Similarly, we can look to see if there are any missing data in the target dataframe

# In[9]:

original_y.isnull().sum()


# Let's see if we have an imbalance in the classification y set

# In[72]:

p = original_y == 'p'
p.sum()


# The data looks complete in both, so now we can convert the string descriptions in the datasets as listed in the "Data Description" section above into integers for use in all of the models. But before we do this, let us copy the 'original' data into the working datasets shrooms_X and shrooms_y to preserve the integrity of the original data.

# In[4]:

shrooms = original.copy()
shrooms_X = original_X.copy()
columns = shrooms_X.columns
shrooms_y = original_y.copy()

for col in columns:
    shrooms_X[col] = shrooms_X[col].astype('category') # the next line relies on a category data type
    shrooms_X[col] = shrooms_X[col].cat.codes # converts categorical data to unique integers
    shrooms_X[col] = shrooms_X[col].astype('int') # convert the column to an integer data type for data exploration

shrooms_y = shrooms_y.astype('category') # the next line relies on a category data type
shrooms_y = shrooms_y.cat.codes # converts categorical data to unique integers
shrooms_y = shrooms_y.astype('int') # convert the column to an integer data type for data exploration


# print(shrooms_X.head())
# print(shrooms_y.head()) # both conversions look good!
shrooms_X.describe()


# #### We also need to create a dataset of categorical dummy attributes in order to separate the various attributes of each characteristic from each other. We will run our models on this set of data as well.

# In[5]:

shrooms_dummy_X = original_X.copy()
columns = shrooms_dummy_X.columns
#columns = columns.drop('class') # we won't make dummy columns for our target column
for col in columns:
    for data in shrooms_dummy_X[col].unique():
        shrooms_dummy_X[col + "_" + data] = shrooms_dummy_X[col] == data
        
# we can now drop the original columns
for col in columns:
    shrooms_dummy_X.drop(col, inplace = True, axis = 1)
#shrooms_dummy_X.head()
# Now let's replace all of the bool values (True/False) with integers (1/0)

shrooms_dummy_X = shrooms_dummy_X.astype(int)
shrooms_dummy_X.head()


# In[13]:

# Create a Pairplot
g = sns.pairplot(shrooms_plot,hue="class",palette="muted",size=5, 
    vars=["odor", "spore-print-color", "stalk-surface-above-ring", "stalk-color-above-ring", "habitat", "population", "cap-color" ],kind='reg')

# To change the size of the scatterpoints in graph
g = g.map_offdiag(plt.scatter,  s=35, alpha=0.5)

# remove the top and right line in graph
sns.despine()
# Additional line to adjust some appearance issue
plt.subplots_adjust(top=0.9)

# Set the Title of the graph from here
g.fig.suptitle('Relation between Primary Mushroom Classification Characteristics', 
    fontsize=60,color="b",alpha=0.3)


# ### The following code defines the four datasets and trains 10 classification models with them. It also collects performance statistics and places all of collected data in a dataframe called results. It takes quite a long time to execute the next frame of code, so the results are in a local CSV file for future access and processing.

# In[24]:

datasets = [
            (shrooms_dummy_X,shrooms_y),
            (shrooms_dummy_X[[
                        'cap-color_w',
                        'odor_a',
                        'odor_l',
                        'odor_n',
                        'stalk-surface-below-ring_y',
                        'stalk-color-above-ring_n',
                        'spore-print-color_r',
                        'population_c',
                        'habitat_l']], shrooms_y),
             (shrooms_X, shrooms_y),               # all characteristics
             (shrooms_X[["odor",                    # only important characteristics for rule-based classification
                        "spore-print-color", 
                        "stalk-surface-above-ring", 
                        "stalk-color-above-ring", 
                        "habitat", 
                        "population", 
                        "cap-color" ]], shrooms_y),
            ]
figure = plt.figure(figsize=(27, 9))

names = [
     "Nearest Neighbors", 
     "Linear SVM", 
     "RBF SVM", 
     "Gaussian Process",
     "Decision Tree", 
     "Random Forest", 
     "Neural Net", 
     "AdaBoost",
     "Naive Bayes",
     "QDA"
    ]

classifiers = [
     KNeighborsClassifier(3, algorithm='auto'),
     GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
     DecisionTreeClassifier(max_depth=5),
     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
     MLPClassifier(alpha=1),
     AdaBoostClassifier(),
     GaussianNB(),
     QuadraticDiscriminantAnalysis()
]

results = pd.DataFrame(columns=('dataset', 'classifier', 'model', 'score', 'conmat', 'roc_auc_score', 'false_positive_rate', 'true_positive_rate'))
labels = ['Poisonous', 'Edible']
i = 0 # counter for adding rows to the results dataset

for dataset_count, dataset in enumerate(datasets):
    
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test =             train_test_split(X, y, test_size=.4, random_state=42)
    

    # iterate over classifiers
    for name, clf in zip(names, classifiers):

        clf.fit(X_train, y_train) 
        y_predict = clf.predict(X_test)
        score = accuracy_score(y_test, y_predict)
        con_mat = confusion_matrix(y_predict, y_test, )
        roc_auc = roc_auc_score(y_test, y_predict)

        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_predict)


         
        results.loc[i] = [dataset_count, name, clf, score, con_mat, roc_auc, false_positive_rate, true_positive_rate]
        i += 1
        

results

# plt.tight_layout()
# plt.show()        


# In[ ]:

#        plt.title('Receiver Operating Characteristic')
#         plt.plot(false_positive_rate, true_positive_rate, 'b',
#         label='AUC = %0.2f'% roc_auc)
#         plt.legend(loc='lower right')
#         plt.plot([0,1],[0,1],'r--')
#         plt.xlim([-0.1,1.2])
#         plt.ylim([-0.1,1.2])
#         plt.ylabel('True Positive Rate')
#         plt.xlabel('False Positive Rate')
#         plt.show()        


# In[79]:

# save the results from above
results.to_csv("results.csv")


# In[26]:

# the plot_learning_curve function will be used as another way to evaluate models
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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, ylim=None, cv='y',
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# ### Below this point is work in progress...

# In[27]:

X, y =             shrooms_dummy_X[[
                        'cap-color_w',
                        'odor_a',
                        'odor_l',
                        'odor_n',
                        'stalk-surface-below-ring_y',
                        'stalk-color-above-ring_n',
                        'spore-print-color_r',
                        'population_c',
                        'habitat_l']], shrooms_y
classifiers = [
     KNeighborsClassifier(3, algorithm='auto'),
     SVC(kernel="linear", C=0.3, probability=True,),
       SVC(gamma=2, C=1),
#      GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
      DecisionTreeClassifier(max_depth=5),
      RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
      MLPClassifier(alpha=1),
      AdaBoostClassifier(),
      GaussianNB(),
      QuadraticDiscriminantAnalysis()
]

names = [
        "Nearest Neighbors", 
        "Linear SVM", 
        "RBF SVM", 
#        "Gaussian Process",
        "Decision Tree", 
        "Random Forest", 
        "Neural Net", 
        "AdaBoost",
        "Naive Bayes",
        "QDA"
    ]

# iterate over classifiers
for name, clf in zip(names, classifiers):
    title = name 
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = clf
    plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
    plt.show()
    
# title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# # SVC is more expensive so we do a lower number of CV iterations:
# cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# estimator = SVC(gamma=0.001)
# plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)
# labels = ['business', 'health']
# cm = confusion_matrix(y_test, pred, labels)
# print(cm)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(cm)
# pl.title('Confusion matrix of the classifier')
# fig.colorbar(cax)
# ax.set_xticklabels([''] + labels)
# ax.set_yticklabels([''] + labels)
# pl.xlabel('Predicted')
# pl.ylabel('True')
# pl.show()


# Some resources for visualization techniques...
# 
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
# http://scikit-learn.org/stable/auto_examples/svm/plot_svm_nonlinear.html
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# http://scikit-learn.org/stable/auto_examples/classification/plot_classification_probability.html#sphx-glr-auto-examples-classification-plot-classification-probability-py
# http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
# 
# https://jmetzen.github.io/2015-01-29/ml_advice.html
# 
# 

# In[ ]:



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


# In[69]:

results['dataset'].astype(int)

for dataset in results['dataset'].unique():
    print("Dataset {}".format(dataset))
    fig = plt.figure()
    fig.tight_layout
    ax1 = fig.add_subplot(111)
    ax1.subtitle('Receiver Operating Characteristic')
    ax1.plot(false_positive_rate, true_positive_rate, 'b',
    label=("Poisonous", "Edible"))
    ax1.legend(loc='lower right')
    ax1.plot([0,1],[0,1],'r--')
#    ax1.xlim([-0.1,1.2])
#    ax1.ylim([-0.1,1.2])
#    ax1.ylabel('True Positive Rate')
#    ax1.xlabel('False Positive Rate')
#    ax1.show()
    
    ax2 = fig.add_subplot(212)
    ax3 = fig.add_subplot(313)
    ax4 = fig.add_subplot(414)
    ax5 = fig.add_subplot(515)
    ax6 = fig.add_subplot(616)
    ax7 = fig.add_subplot(717)
    ax8 = fig.add_subplot(818)
    ax9 = fig.add_subplot(919)
    plt.show
    plt.clf


# In[42]:

shrooms_plot = shrooms_dummy_X[[
                        'cap-color_w',
                        'odor_a',
                        'odor_l',
                        'odor_n',
                        'stalk-surface-below-ring_y',
                        'stalk-color-above-ring_n',
                        'spore-print-color_r',
                        'population_c',
                        'habitat_l']]

shrooms_plot.loc[:,'class'] = shrooms_y
g = sns.pairplot(shrooms_plot)


# In[ ]:




# In[ ]:



