#!/usr/bin/env python

import sys
import re

def clean_text():
    for line in sys.stdin:
        line = line.lower()
        print line

if __name__ == '__main__':
    clean_text()
