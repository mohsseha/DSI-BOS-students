import sys
inFile = sys.argv[1]

def word_counter(word_list):
    word_dict = {}
    for i in word_list:
        if i in word_dict.keys():
            word_dict[i] += 1
        else:
            word_dict[i] = 1
    for i in word_dict:
        print(i, word_dict[i])
    return(word_dict)

def text_file_word_splitter(textfile):
    import re
    f = open(textfile,'r',encoding='utf8')
    x = f.read()
    x = re.sub(r'[^a-zA-Z0-9 ]', '', x)
    x = x.lower()
    word_list = x.split()
    word_counter(word_list)

text_file_word_splitter(inFile)
