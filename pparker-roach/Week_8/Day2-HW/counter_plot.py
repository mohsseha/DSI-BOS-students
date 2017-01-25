
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
    word_count_dictionary = {}                         # create a dictionary to hold word counts

    for line in sys.stdin:                             # get a line of input from input stream
        line = my_cleaner.clean_text(line)
        split_line = line.split(" ")                   # break the line of input into segments at spaces
        for word in split_line:                        # for each word
            #update dictionary with stripped_word
            if word in word_count_dictionary.keys():
                word_count_dictionary[word] += 1
            else:
                word_count_dictionary[word] = 1

    for key, item in word_count_dictionary.items():
        word_count = key + ' = ' + str(item) + '\n'
        sys.stdout.write(word_count)
    #sorted_dic = OrderedDict(sorted(word_count_dictionary.items(), key=lambda x:x[1], reverse=True))
    LABELS = list(word_count_dictionary)
    VALUES = list(word_count_dictionary.values())
    plt.hist(VALUES, bins = 20)
    plt.show()



# In[ ]:




# In[ ]:
