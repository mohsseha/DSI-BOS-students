#!/usr/bin/python   

from sys import stdin

def word_counter ():
    
    punctuation = [",", ".", "!", "?", # pauses and stops
                   " '", "' ", '"', u"\u201D", u"\u201C", # quotations
                   "(", ")", "[", "]", "{", "}", # parenthases, brackets and braces
                   ";", ":", # colons
                   "\n", "\t", # white spaces
                   " -", u"\u2013", u"\u2014", # hyphens and dashes
                   "*", "/", "#" # miscellaneous
                  ]
    
    count_dictionary = {}
    
    # Get text from stdin
    for line in stdin.readlines():
        

        # Split the text string according to punctuation, and remove any double spaces
        string_list = line
        for splitter in punctuation:
            string_list = string_list.replace(splitter, " ").replace("  ", " ")

        # Split the string by spaces
        split_string_list = string_list.split(" ")

        # Count word frequency
        for word in split_string_list:
            if word.lower() in count_dictionary.keys():
                count_dictionary[word.lower()] += 1
            else:
                count_dictionary[word.lower()] = 1
    
    # Remove blank entries from the count dictionary
    if "" in count_dictionary.values():
        del count_dictionary[""]

    # Print the dictionary
    for word in sorted(count_dictionary.keys()):
        print("{}\t{}".format(word, count_dictionary[word]))

#    with open("./counter_output.txt", 'w') as o:
#        for word in sorted(count_dictionary.keys()):
#            o.write("{}\t{}\n".format(word, count_dictionary[word]))
#            
if __name__ == '__main__':

    word_counter()
