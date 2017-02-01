
# coding: utf-8

# # Setting up Foursquare data for analysis 

# 
# > Note: This will be a very open ended lab, since everyone may end up using different geographies and starting seed geographies. Be prepared to walk around and hand-hold some people, I've tested this out on several locales around me and it works, for most, but if you don't have a good starting seed location, the procedure may not scrape well.
# 
# Today's lab is going to get your hands dirty with respect to the Foursquare API. We're also going to build a simple crawler/scraper that will go through the JSON hierarchy, extract the data we want, and deposit them into a Pandas table so we can do simple analysis. 
# 
# Just in case you're unfamiliar with this concept, please refer to the Wikipedia page (it's actually pretty good): https://en.wikipedia.org/wiki/Web_scraping, and maybe spend a few moments discussing the concepts and how it could help you in the future as a data scientist to have this "hackish" skill. 

# Setup your access token to foursquare

# In[2]:

# Solutions

import foursquare
import json
import pandas as pd
import unicodedata


#ACCESS_TOKEN = ""
#client = foursquare.Foursquare(access_token=ACCESS_TOKEN)

CLIENT_ID = 'YOUR CODE HERE'
CLIENT_SECRET = 'YOUR CODE HERE'
client = foursquare.Foursquare(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)


# Use a foursquare python library method to search for suitable venues around a city near you. Print the associated JSON output in a nice way with appropriate spacing and indentation

# In[4]:

# Solution

starting_list = client.venues.search(params={'near': 'Blah, GA', 'radius':'1500'})
print(starting_list['venue'][0])


# Wow... that should look like a total mess to you. Read the following docs: https://docs.python.org/2/library/json.html, and read the part about pretty printing. Once you think you've understood the method, deploy it here and see the world a difference a bit of spacing and indenting makes! 

# In[17]:

print(json.dumps(starting_list, indent = 4))


# Now that we can make some sense of the structure let's practice traversing the JSON hieararchy, select one of the venues in the list and output it's name

# In[14]:

# Solution
type(starting_list['venues'][23]['categories'][0]['name'])


# Note that the output isn't exactly what we want. It says u'Park', and if you check the type, Python will output Unicode. This isn't good, we need to recover the original intended type. Read the following docs: 
# 
# https://docs.python.org/2/library/unicodedata.html, and checkup the method 'normalize'. Once you think you've understood this method. Implement it on the above call and see if you can recover the appropriate type for that data.
# 

# Now for some exploratory analysis, let's print the number of total venues in your list

# In[4]:

# Solution

len(starting_list['venues'])


# Extract the location id for your starting list. Make sure it's normalized to its correct type, and not Unicode. Put this id in a variable called temp. From this id, we will get a list of other venues.

# In[13]:

# Solution

temp = unicodedata.normalize('NFKD', starting_list['venues'][17]['id']).encode('ascii','ignore')


# Print the venues list (in the nicely formatted JSON)

# In[6]:

# Solution

temp1 = client.venues(temp);
print(json.dumps(temp1, indent = 4))


# Create a procedure that will only extract the comments in a list. There are a few ways you can do this, but I highly recommend you look up the map method from the base Python library: https://docs.python.org/2/tutorial/datastructures.html
# 
# This is the same "map" function, that's one part of the map-reduce duo used in "Big Data" applications. So it may be helpful to get familiar with this method now if that's where you think you may want to take your career in the future. 

# In[8]:

# Solution
map(lambda h: h['text'], temp1['venue']['tips']['groups'][0]['items'])


# Now we're going to bring the above mini-tasks together into a nice little method, that will allow us to convert any foursquare JSON data into a nice tabular / rectangular table for further analysis. First instnatiate a pandas data frame.

# In[9]:

venue_table = pd.DataFrame()


# Write a procedure that will take your list of venues around a certain geography/lat/long whatever, and output a table that will have for each row, a comment associated for the venue (multiple comments will mean multiple rows, each per comment), the venue name, the tip count, the user count, and the store category. Make sure that each column is populated with appropriately typed values, i.e. names/categories should be strings, and numbers should be numerical data type.

# > To the instructor: I usually don't have this much latitude to the student, but it was requested that I give some "open ended"/"munch on" problems. I suspect the students will spend the most time here, they will certainly get errors, and they will be frustrated. Look through the ideal solution and be prepared to step in when appropriate. 
# 
# **Hint**: Before you begin, think about the process. You're going to start with a loop of some kind, then think about the following:
# - How many of those do you need? 
# - Think about the JSON structure, how "deep" do you need to penetrate the hierarchy to reach the data you want (this will help you think about how many loops you need for your crawler
# - How should you iteratively add on to your Pandas data frame? 
# - Think of any tests you may need to put in to ensure your procedure does not cause an error (this may help you figure out how many if statements you may need, and where to place them.
# 

# In[10]:

# Solution - Note to instructor, the code may be slightly different, in particular the student should have written error-exception protocols to account for any 
# missing/empty values that may cause the procedure to kick-out in an error.

for v_index in range(len(starting_list['venues'])-1):
    temp = unicodedata.normalize('NFKD', starting_list['venues'][v_index]['id']).encode('ascii','ignore')
    temp1 = client.venues(temp)
    print v_index
    comment_list = map(lambda h: h['text'], temp1['venue']['tips']['groups'][0]['items'])
    for c_index in range(len(comment_list)-1):
        print c_index
        comment_converter = unicodedata.normalize('NFKD', comment_list[c_index]).encode('ascii','ignore')
        print "test"
        if (starting_list['venues'][v_index]['categories']) != []:  
            venue_table = venue_table.append(pd.DataFrame({"name": unicodedata.normalize('NFKD', starting_list['venues'][v_index]['name']).encode('ascii','ignore'),
                                            "tip count": starting_list['venues'][v_index]['stats']['tipCount'],
                                            "users count": starting_list['venues'][v_index]['stats']['usersCount'],
                                             "store category": unicodedata.normalize('NFKD', starting_list['venues'][v_index]['categories'][0]['name']).encode('ascii','ignore'), 
                                             "comments": comment_converter}, index = [v_index + c_index]))
        else:
            venue_table = venue_table.append(pd.DataFrame({"name": unicodedata.normalize('NFKD', starting_list['venues'][v_index]['name']).encode('ascii','ignore'),
                                            "tip count": starting_list['venues'][v_index]['stats']['tipCount'],
                                            "users count": starting_list['venues'][v_index]['stats']['usersCount'],
                                             "store category": "No categories", 
                                             "comments": comment_converter}, index = [v_index + c_index]))


# Finally, output the Venue table

# In[11]:

venue_table.drop_duplicates()


# You've done it! You've built a simple crawler that traverses a JSON directory, and you've deposited the results in a nice Pandas data frame. Congratulations! You're now ready for more data-mining in the future, and have just beefed up the **data** part of the data science combination :)

# In[ ]:



