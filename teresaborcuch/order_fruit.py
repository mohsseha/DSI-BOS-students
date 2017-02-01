#!/usr/bin/python

import sys
import string

fruit_list = []

for line in sys.stdin:
    line = line.lower()
    fruit_list.append(line)
    fruit_list = sorted(fruit_list)

f = open('./fruit_output.txt', "w")
for word in fruit_list:
    f.write(word)
