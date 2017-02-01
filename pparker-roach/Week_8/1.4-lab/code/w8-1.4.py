
# coding: utf-8

# # Fun time with Bayesian statistics and Politics 

# In[1]:

from IPython.display import Image
Image(url='http://a.abcnews.go.com/images/Politics/GTY_donald_trump_hillary_clinton_sk_150619_16x9_992.jpg')


# The presidential voting season is upon us, and what better way to celebrate it, and our newfound understanding of Bayesian statistics by fusing the two together (it works for Asian food), and of course, this is not without precedent. In the last presidential electoral cycle, Nate Silver won acclaim (and some scorn from academia for the acclaim he garnered with work that lacked much novelty) for utilizing a Bayesian approach to his poll prediction, in particular he utilized something called a Monte Carlo Markov Chain (MCMC). We're going to start our nascent political data science careers in this lab with something slightly less ambitious, but no less relevant to the material we've gone through so far.

# # Data Background 

# # Some model background

# #### 2.1 Priors and Conjugate Priors

# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Conjugate Priors
# | Prior  | Likelihood  | 
# |---|---|
#  | Beta  | Binomial  |
#  | Dirichlet  | Multinomial   | 
#  | Gamma  | Gaussian |
#  | Gaussian  | Gaussian | 
#  
#  The above table provides the likelihood/prior conjugate prior combinations that ensure computational tractability for calculation of posterior distributions 

# We've seen some of these previously, but let's just summarize these distributions now: 
# 
# | Name  | Function (The Distribution/PMF) - Whatever| Use
# |---|---|
# | Beta  | $\frac{1}{B(\alpha, \beta)}x^{\alpha-1}(1-x)^{\beta-1}$ | More of a family of distributions, usually representing probabilties of proportions, thus it's great to use as our prior belief (very important in Bayesian analysis)|
# | Dirichlet  |  $\begin{equation*}\frac{1}{B(\alpha, ,\beta)}\prod_{i = 1}^{n}x_i^{\alpha-1}\end{equation*}$  | Similar to Beta above, just extended over multinomials, often used in NLP and other text-mining processes  |
# | Gamma  |$ x^{a-i}e^{-x}\frac{1}{\Gamma(a)}$ | This distribution represents waiting time between Poisson distributed events (remember c_i in the previous lecture finger exercise | 
# | Gaussian  | $\frac{1}{\sigma \sqrt(2\pi)}e^{\frac{-(x-\mu)^{2}}{2\sigma^2}}$ | This is just the normal distribution | 
# | Binomial  | $\binom{n}{k}\cdot p^kq^{n-k} $   | Gives you the probability of getting k "success" in n iterations/trials |
# | Multinomial  | $\frac{N!}{\prod_{i=1}^n x_i!} \prod_{i=1}^n \theta_i^{x_i} $| Similar to binomial distribution, but generalized to include multiple "buckets" (X, Y, Z,...) instead of just two (1 or 0), represents how many times each bucket X, Y, Z,... etc was observed over n trials | 

# #### 2.2 Beta functions as normalizing term in case you don't want to make your life easy and use conjugate priors...

# We are finally going to use the much anticipated Beta Function today to do our first real computational Bayesian exercise, $\frac{1}{B(\alpha, \beta)}x^{\alpha-1}(1-x)^{\beta-1}$. Wait, what's B? We can define B as the following: 
# $$B(a,b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$$
# 
# As it is used in Bayesian Analysis, the beta distribution (not "B" - or the beta function) is often used to represent the prior beliefs of the modeler. How so? Well let's say you thought the chance of a political candidate winning an election was .60 (whatever that means), but that given new information, that you've cleverly modeled using Bayes, it's now .80 (say you've tied your candidates political fortunes to the performance of the economy), your prior beliefs can be modeled (after some tuning of $\alpha$, $\beta$, the input variable) by the beta distribution.
# 
# **The beta function** in contrast, is often used as a scaling term that ensures your model is bounded by (0, 1).
#  

# Build the Beta function into Python using the math library

# In[1]:

import math
from scipy.special import gamma as Gamma
from scipy.stats import beta


# #### 2.3 (An advanced preview) Introducing Dirchilet Distribution for Multi-level Categorical Analysis

# In the longer term, our main aim is to model the quotient, in particular, the percentage of voters for a particular candidate. The 'commonest' way to do this (to borrow a phrase from our British friends), is to utilize the Dirchilet distribution in our probability model: $\begin{equation*}\frac{1}{B(\alpha)}\prod_{i = 1}^{n}x_i^{\alpha-1}\end{equation*}$ (stay tuned for this later)  
# 
# As you read in the summary in the table above, it's often used to model situations with more than "2" possiblities/buckets/outcomes/whatever. This is in fact a very powerful distribution. 
# 
# For our purposes, we would use this distribution to model situations where there are more than 2 candidates (Clinton, Bush, Perot in 1992, or maybe Clinton, Trump, Sanders in 2016. But for this lab,  let's assume there's just two alternatives. That way, we can utilize the simple binomial model and as our table states, we just need to use beta for that case.
# 

# # Prepping the data

# This case study will be based off of the following data: http://www.stat.columbia.edu/~gelman/arm/examples/election88/, from the **great one**, Dr. Gelman, one of the foremost researchers in Bayesian analysis. 
# 
# This election was ultimately a major victory for the Grand Old Party (GOP/Republicans), and propelled George H.W. Bush (the father of George W. Bush) into the presidency. 
# 
# Import the data, and delete the Unnamed column

# In[3]:

import pandas as pd

# Reading in the csv file


# Calling a conversion method of your data to csv 



# The first column "Unnamed" is clearly just a row-identifer machine generated. We can ignore that column or drop it all together



# Further, we can remove org, as it is just one value -- Special note, in general, we probably don't want to do this, because 
# there may be a chance we want to merge/join this file with another and that column could end up being a unique identifier for 
# a subset of data in our large dataset



# Now that we have the data read in, we would like to partition the data into states so we can do state-specific inference 

# Summarize the age variable with counts

# In[2]:

# Determine the number of yers of the data


# Summarize the state with counts

# In[4]:

# Determine the number of entities/states of the data


# cretae a data frame with only state and age

# In[5]:

# We need to partition the data into individual data frames to construct our Bayesian inference on 


# In[ ]:




# Great, now let's do some basic exploratory analysis on the polling data. Let's construct the Bush support for state-age cohorts, and save that data for use in the dictionary, container.

# In[6]:

import numpy as np
container = {}; bush_quotient = 0


            
# Likewise, let's also produce a dictionary with the general poll percentages for state (no age)



# You can also probably do something similar to the method above by utilizing the groupby method in Pandas
# 
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html
# 
# If you'd like extra practice, please go ahead and recreate the above with this simple pandas method

# In[15]:

# Write your optional procedure here


# Let's just check out what # 1 shows (all ages)

# In[ ]:




# In[7]:

# We also have the actual Election 88 sample percentages to compare with the polling. Let's load the data just to have a look
import pandas as pd



# So the electoral data shows that the actual results were .59, whereas the polling showed .71. Does that mean Bush somehow lost almost 17% of his support from the time of the poll to the election? We can put some clarify on this question by doing a simple Bayesian computation

# # Combining The Prior and Likelihood 

# We see from the table way back at the top that using Beta-Binomial Prior-Likelihood combination allows us to remove the denominator/scaling term, and makes the right-hand side of the Bayesian formula analytically tractable (we can use a Riemann or 'normal' Integral or a simple summation) to compute the probability. Wonderful, polling data is naturally contextualized using Binomial distribution! 
# 
# So let's test whether or not $\theta \le .65$, that would basically be the following closed form equation : 
# $$P(\theta \le .65) = \frac{\Gamma( 404 )}{\Gamma(141)\Gamma (263)} \int_{0}^{.65} \theta^{144}(1-\theta)^{59}\theta^{119}(1-\theta)^{82}$$
# 
# Note: A closed form equation is just a fancy way to mean you can write write something as A = **blah**, where **blah** is a formula that can be computed and is not infinite. 
# 

# In[ ]:




# In[8]:

import scipy.integrate as integrate
from scipy.stats import beta




# Uh oh, doesn't work.... what happened? Well, the gamma funciton is not computaitonally tractable. We'll need to figure out, A different way to compute Beta..... what can that be? 
# 
# What does tractability mean in data science/computer science? First let's look deeper at my_beta above. Note we use the method gamma to construct beta. This method refers to the gamma **function** and not the distribution. What is gamma? 
# 
# Please read the following:
# http://mathworld.wolfram.com/GammaFunction.html
# 
# Note $\Gamma(a)=(a-1)!$, and if you recall (a-1)! = (a-1)(a-2)...(2)(1). For fun, try to calculate $\Gamma(90)$, $\Gamma(100)$, anything funny happen? Overflow you say? Yep. Gamma explodes very quickly. This is bad for computers (or more precisely Von-Neumann computers -which we all currently use). 
# 
# Therefore, we need to find some trick around this explosion. What about logs? Take My_beta and take the log (something you should have had some practice with so far). 

# In[ ]:




# In[ ]:




# # Conclusion and Towards The Future

# We see that the $P(\theta \le .65)$ is very close to 0, and therefore we believe the poll did not provide "good" information with respect to the actual electoral results 
