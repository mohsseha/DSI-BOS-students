
# coding: utf-8

# In[ ]:

import sys
import string


# In[ ]:

if __name__ == "__main__":
    word_count_dictionary = {}                         # create a dictionary to hold word counts
    punctuation = string.punctuation
    
    for line in sys.stdin:                             # get a line of input from input strea,
        split_line = line.split(" ")                   # break the line of input into segments at spaces
        for word in split_line:                        # for each word
            #strip punctuation
            translator = str.maketrans('', '', string.punctuation)
            stripped_word = (word.translate(translator))
            #update dictionary with stripped_word
            if stripped_word in word_count_dictionary.keys():
                word_count_dictionary[stripped_word] += 1
            else:
                word_count_dictionary[stripped_word] = 1

for key, item in word_count_dictionary.items():
    word_count = key + ' = ' + str(item) + '\n'
    sys.stdout.write(word_count)


# In[ ]:



