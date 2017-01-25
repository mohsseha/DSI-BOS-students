

if __name__ == "__main__":
    import sys
    import my_cleaner
    import matplotlib.pyplot as plt
    word_dict = {}
    for line in sys.stdin:
      line = my_cleaner.clean_text(line)
      for i in line.split():
          if i in word_dict.keys():
              word_dict[i] += 1
          else:
              word_dict[i] = 1
    for i in word_dict:
        print(i, word_dict[i])

    #X = range(len(word_dict))
    #y = word_dict.values
    #print(X,y)
    #top_5 = sorted(word_dict.values(), reverse=True)[0:5]
    #print(top_5)
    plt.bar(range(len(word_dict)),word_dict.values(), align='center')
    plt.xticks(range(len(word_dict)),word_dict.keys())
    plt.show()
    #for i in top_5:
    #    print(i, word_dict[i])
