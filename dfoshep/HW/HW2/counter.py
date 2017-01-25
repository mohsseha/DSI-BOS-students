#!/usr/bin/env python

import sys
import string
import my_cleaner
from collections import defaultdict

wordcounts = defaultdict(int)

for line in sys.stdin:
    words = my_cleaner.clean_text(line)

    for word in words:
        wordcounts[word] += 1


for word, count in sorted(wordcounts.items()):

    print "{}  {}".format(word,count)
