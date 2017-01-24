import sys
import re

"""
Prints out the unique words in a .txt file along with the word count. Will not
work with a .csv file since the function removes ',' values.
"""

word_dict = {} # initialize a blank dictionary
for line in sys.stdin:
    line = re.sub('[!@#$,.;]', '', line) # removes punctuation
    line = re.sub('\xe2.*', '', line) # removes unicode issues
    line = line.strip()
    if not line: # if line is blank, then pass
        pass
    else:
        for word in line.split():
            # checks if the word is already in the dictionary
            # if it is then add one to the value count
            # else set the value count at 1
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1

# iterate over the key:value pairs in the dictionary and print both
# with a tab seperating the values
for key in word_dict.keys():
    print "{} \t {}".format(key, word_dict[key])
