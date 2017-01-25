#!/usr/bin/python

import sys
import string

# function takes in a line of text and cleans it
def clean_text(x):
    x = x.encode('ascii')
    x = x.lower()
    x = x.translate(None, string.punctuation)
    x = x.strip()
    text_words = x.split()

    return text_words
