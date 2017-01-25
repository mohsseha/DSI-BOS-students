#!/usr/bin/env python

import sys
import my_cleaner

word_dict = {}

for line in sys.stdin:
    words = my_cleaner.clean_text(line)

    for word in words:
        if word not in word_dict:
            word_dict[word] = 1
        else:
            word_dict[word] += 1

for word in word_dict:
    print word + ' ' + str(word_dict[word])
