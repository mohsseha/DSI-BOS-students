if __name__ = "__main__":
    import sys
    from text_clean import clean_text
    for line in sys.stdin:
        line = clean_text(line)
        for word in line:
                if word not in word_dict.keys():
                    word_dict[word] = 1 #if a word isn't in the dictionary, add it
                else:
                    word_dict[word] +=1 #otherwise add one to the count
    for key in word_dict.keys():
        print key,    word_dict[key] #print word and count pairs
