#!/usr/bin/env python

import sys
import re

def clean_text():
    for line in sys.stdin:
        line = re.sub('[!@#$,.;?":]', '', line) # removes punctuation
        line = re.sub('\xe2.*', '', line) # removes unicode issues
        line = line.strip().lower()
        print line

if __name__ == '__main__':
    clean_text()
