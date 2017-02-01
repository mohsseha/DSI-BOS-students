
# coding: utf-8

# This program only cleans the text, and returns the text (which can be fed into `counter`

# In[92]:

import re
from sys import stdin

def count_words ():
    text_string = DEFAULT
    text_string = re.sub("[\[\]\(\)\"*?!~`.,;:]", "", text_string)
    text_string = re.sub("[' ][ ']", " ", text_string)
    print(text_string)
    


# In[93]:

DEFAULT = """Either escapes' special characters (permitting you to match characters like '*', '?', and so forth), or signals a special sequence; special sequences are discussed below.

If you’re not using a raw string to express the pattern, remember that Python also uses the backslash as an escape sequence in string literals; if the escape sequence isn’t recognized by Python’s parser, the backslash and subsequent character are included in the resulting string. However, if Python would recognize the resulting sequence, the backslash should be repeated twice. This is complicated and hard to understand, so it’s highly recommended that you use raw strings for all but the simplest expressions."""


# In[94]:

count_words()


# In[ ]:



