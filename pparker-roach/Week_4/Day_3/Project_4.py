
# coding: utf-8

# # Web Scraping for Indeed.com & Predicting Salaries

# In this project, we will practice two major skills: collecting data by scraping a website and then building a binary predictor with Logistic Regression.
# 
# We are going to collect salary information on data science jobs in a variety of markets. Then using the location, title and summary of the job we will attempt to predict the salary of the job. For job posting sites, this would be extraordinarily useful. While most listings DO NOT come with salary information (as you will see in this exercise), being to able extrapolate or predict the expected salaries from other listings can help guide negotiations.
# 
# Normally, we could use regression for this task; however, we will convert this problem into classification and use Logistic Regression.
# 
# - Question: Why would we want this to be a classification problem?
# - Answer: While more precision may be better, there is a fair amount of natural variance in job salaries - predicting a range be may be useful.
# 
# Therefore, the first part of the assignment will be focused on scraping Indeed.com. In the second, we'll focus on using listings with salary information to build a model and predict additional salaries.

# ### Scraping job listings from Indeed.com

# We will be scraping job listings from Indeed.com using BeautifulSoup. Luckily, Indeed.com is a simple text page where we can easily find relevant entries.
# 
# First, look at the source of an Indeed.com page: (http://www.indeed.com/jobs?q=data+scientist+%2420%2C000&l=New+York&start=0&limit=3000")
# 
# Notice, each job listing is underneath a `div` tag with a class name of `result`. We can use BeautifulSoup to extract those. 

# #### Setup a request (using `requests`) to the URL below. Use BeautifulSoup to parse the page and extract all results (HINT: Look for div tags with class name result)
# 
# The URL here has many query parameters
# 
# - `q` for the job search
# - This is followed by "+20,000" to return results with salaries (or expected salaries >$20,000)
# - `l` for a location 
# - `start` for what result number to start on

# In[86]:

URL = "http://www.indeed.com/jobs?q=data+scientist+%2420%2C000&l=New+York&start=0&limit=100"


# In[87]:

import requests
import bs4
from bs4 import BeautifulSoup


# In[90]:

r = requests.get(URL)

soup = BeautifulSoup(r.content)
#soup.find('div',{"class" : "result"})



# Let's look at one result more closely. A single `result` looks like
# 
# ```
# <div class=" row result" data-jk="2480d203f7e97210" data-tn-component="organicJob" id="p_2480d203f7e97210" itemscope="" itemtype="http://schema.org/JobPosting">
# <h2 class="jobtitle" id="jl_2480d203f7e97210">
# <a class="turnstileLink" data-tn-element="jobTitle" onmousedown="return rclk(this,jobmap[0],1);" rel="nofollow" target="_blank" title="AVP/Quantitative Analyst">AVP/Quantitative Analyst</a>
# </h2>
# <span class="company" itemprop="hiringOrganization" itemtype="http://schema.org/Organization">
# <span itemprop="name">
# <a href="/cmp/Alliancebernstein?from=SERP&amp;campaignid=serp-linkcompanyname&amp;fromjk=2480d203f7e97210&amp;jcid=b374f2a780e04789" target="_blank">
#     AllianceBernstein</a></span>
# </span>
# <tr>
# <td class="snip">
# <nobr>$117,500 - $127,500 a year</nobr>
# <div>
# <span class="summary" itemprop="description">
# C onduct quantitative and statistical research as well as portfolio management for various investment portfolios. Collaborate with Quantitative Analysts and</span>
# </div>
# </div>
# </td>
# </tr>
# </table>
# </div>
# ```
# 
# While this has some more verbose elements removed, we can see that there is some structure to the above:
# - The salary is available in a `nobr` element inside of a `td` element with `class='snip`.
# - The title of a job is in a link with class set to `jobtitle` and a `data-tn-element="jobTitle`.  
# - The location is set in a `span` with `class='location'`. 
# - The company is set in a `span` with `class='company'`. 

# ### Write 4 functions to extract each item: location, company, job, and salary.
# 
# example: 
# ```python
# def extract_location_from_result(result):
#     return result.find ...
# ```
# 
# 
# - **Make sure these functions are robust and can handle cases where the data/field may not be available.**
#     - Remember to check if a field is empty or `None` for attempting to call methods on it
#     - Remember to use `try/except` if you anticipate errors
# - **Test** the functions on the results above and simple examples
# 
# results = soup.find_all('div', attrs={'data-tn-component': 'organicJob'})
# 
# for x in results:
#    company = x.find('span', attrs={"itemprop":"name"})
#    print('company:', company.text.strip())
# 
#    job = x.find('a', attrs={'data-tn-element': "jobTitle"})
#    print('job:', job.text.strip())
# 
#    salary = x.find('nobr')
#    if salary:
#        print('salary:', salary.text.strip())

# In[92]:

#results = soup.find_all('div', attrs={'data-tn-component': 'organicJob'})
#results = soup.find_all('td', attrs={"class":"jobtitle turnstilelink"})



def link_filter(new_url):
    if 'rc/clk' in new_url:
        pass
    else:
        job_title = getTitle(new_url)
        return job_title

def getTitle(URL):
    r = requests.get(URL)
    soup = BeautifulSoup(r.content, 'html.parser')
    jobtitle = soup.find('b', {'class':'jobtitle'})
    title = jobtitle.text.strip()
    return title

results = soup.find_all('div', attrs={'data-tn-component': 'organicJob'})
ID = []
Title = []
for x in results:
    company = x.find('span', attrs={"itemprop":"name"})
    job = x.find('a', attrs={'data-tn-element': "jobTitle"})
  
    job_id = x.find('h2', attrs={"class": "jobtitle"})['id']
    ID.append(job_id)
  
    #Link
    link = x.find('h2', attrs={"class": "jobtitle"}).find('a')['href']
    job_url = "https://www.indeed.com" + link

    job_link = BeautifulSoup(r.content, 'html.parser')

    Title.append(link_filter(job_url))
print (ID)
print (Title)
indeed = {'ID': ID}
print(indeed)

indeed = {'ID': ID, 'Title':Title}
print(indeed)


df = pd.DataFrame(indeed, columns=["ID","Title"])
print(df)


# Now, to scale up our scraping, we need to accumulate more results. We can do this by examining the URL above.
# 
# - "http://www.indeed.com/jobs?q=data+scientist+%2420%2C000&l=New+York&start=10"
# 
# There are two query parameters here we can alter to collect more results, the `l=New+York` and the `start=10`. The first controls the location of the results (so we can try a different city). The second controls where in the results to start and gives 10 results (thus, we can keep incrementing by 10 to go further in the list).

# #### Complete the following code to collect results from multiple cities and starting points. 
# - Enter your city below to add it to the search
# - Remember to convert your salary to U.S. Dollars to match the other cities if the currency is different

# In[5]:

YOUR_CITY = ''


# In[6]:

url_template = "http://www.indeed.com/jobs?q=data+scientist+%2420%2C000&l={}&start={}"
max_results_per_city = 100 # Set this to a high-value (5000) to generate more results. deed
        # Append to the full set of results
        pass


# #### Use the functions you wrote above to parse out the 4 fields - location, title, company and salary. Create a dataframe from the results with those 4 columns.

# In[7]:

## YOUR CODE HERE


# Lastly, we need to clean up salary data. 
# 
# 1. Only a small number of the scraped results have salary information - only these will be used for modeling.
# 1. Some of the salaries are not yearly but hourly or weekly, these will not be useful to us for now
# 1. Some of the entries may be duplicated
# 1. The salaries are given as text and usually with ranges.
# 
# #### Find the entries with annual salary entries, by filtering the entries without salaries or salaries that are not yearly (filter those that refer to hour or week). Also, remove duplicate entries

# In[9]:

## YOUR CODE HERE


# #### Write a function that takes a salary string and converts it to a number, averaging a salary range if necessary

# In[10]:

## YOUR CODE HERE


# ### Save your results as a CSV

# In[14]:

## YOUR CODE HERE


# ## Predicting salaries using Logistic Regression

# #### Load in the the data of scraped salaries

# In[12]:

## YOUR CODE HERE


# #### We want to predict a binary variable - whether the salary was low or high. Compute the median salary and create a new binary variable that is true when the salary is high (above the median)
# 
# We could also perform Linear Regression (or any regression) to predict the salary value here. Instead, we are going to convert this into a _binary_ classification problem, by predicting two classes, HIGH vs LOW salary.
# 
# While performing regression may be better, performing classification may help remove some of the noise of the extreme salaries. We don't have to choice the `median` as the splitting point - we could also split on the 75th percentile or any other reasonable breaking point.
# 
# In fact, the ideal scenario may be to predict many levels of salaries, 

# In[15]:

## YOUR CODE HERE


# #### Thought experiment: What is the baseline accuracy for this model?

# In[16]:

## YOUR CODE HERE


# #### Create a Logistic Regression model to predict High/Low salary using statsmodel. Start by ONLY using the location as a feature. Display the coefficients and write a short summary of what they mean.

# In[17]:

## YOUR CODE HERE


# #### Create a few new variables in your dataframe to represent interesting features of a job title.
# - For example, create a feature that represents whether 'Senior' is in the title 
# - or whether 'Manager' is in the title. 
# - Then build a new Logistic Regression model with these features. Do they add any value? 
# 

# In[18]:

## YOUR CODE HERE


# #### Rebuild this model with scikit-learn.
# - You can either create the dummy features manually or use the `dmatrix` function from `patsy`
# - Remember to scale the feature variables as well!
# 

# In[19]:

## YOUR CODE HERE


# #### Use cross-validation in scikit-learn to evaluate the model above. 
# - Evaluate the accuracy, AUC, precision and recall of the model. 
# - Discuss the differences and explain when you want a high-recall or a high-precision model in this scenario.

# In[20]:

## YOUR CODE HERE


# #### Compare L1 and L2 regularization for this logistic regression model. What effect does this have on the coefficients learned?

# In[21]:

## YOUR CODE HERE


# In[22]:

## YOUR CODE HERE


# #### Continue to incorporate other text features from the title or summary that you believe will predict the salary and examine their coefficients

# #### Take ~100 scraped entries with salaries. Convert them to use with your model and predict the salary - which entries have the highest predicted salaries?

# ### BONUS 

# #### Bonus: Use Count Vectorizer from scikit-learn to create features from the text summaries. 
# - Examine using count or binary features in the model
# - Re-evaluate the logistic regression model using these. Does this improve the model performance? 
# - What text features are the most valuable? 

# In[23]:

## YOUR CODE HERE


# In[24]:

## YOUR CODE HERE


# #### Re-test L1 and L2 regularization. You can use LogisticRegressionCV to find the optimal reguarlization parameters. 
# - Re-test what text features are most valuable.  
# - How do L1 and L2 change the coefficients?

# In[25]:

## YOUR CODE HERE


# In[ ]:



