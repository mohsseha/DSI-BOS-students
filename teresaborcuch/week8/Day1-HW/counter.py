#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import string

word_list = []
for line in sys.stdin:
    line = line.lower()
    line = line.translate(None, string.punctuation)
    line = line.translate(None, '—')
    line = line.translate(None, '“')
    line = line.translate(None, "‘")
    line = line.translate(None, "\x9d")
    line_list = line.split()
    word_list = word_list + line_list

word_dict = {}
for word in word_list:
    if word not in word_dict:
        word_dict[word] = 1
    else:
        word_dict[word] = word_dict[word] + 1

f = open('./output.txt', "w+")

for word in word_dict:
    string = (word + ' '+ str(word_dict[word])+'\n')
    f.write(string)
