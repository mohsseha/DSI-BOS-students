# coding: utf-8

# In[27]:

import sys
import string
import my_cleaner
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict


# In[52]:

if __name__ == "__main__":
    word_count_dictionary = OrderedDict()                        # create a dictionary to hold word counts
    x = OrderedDict()                        # create a dictionary to hold word counts

    for line in sys.stdin:                             # get a line of input from input stream
        line = my_cleaner.clean_text(line)
        split_line = line.split(" ")                   # break the line of input into segments at spaces
        for word in split_line:                        # for each word
            #update dictionary with stripped_word
            if word in word_count_dictionary.keys():
                word_count_dictionary[word] += 1
            else:
                word_count_dictionary[word] = 1

    x = sorted(word_count_dictionary)
    for  item in x:
        fruit = str(item) + '\n'
        sys.stdout.write(fruit)




# In[ ]:




# In[ ]:
