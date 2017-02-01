#!/usr/bin/env python

import sys
import my_cleaner


def word_counter():
    word_dict = {}
    for line in sys.stdin:
        line = my_cleaner.clean_text()
        if not line: # if line is blank, then pass
            pass
        else:
            # checks if the word is already in the dictionary
            # if it is then add one to the value count
            # else set the value count at 1
            for word in line.split(" "):
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
    print word_dict


if __name__ == '__main__':
    word_counter()
