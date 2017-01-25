if __name__ == "__main__":
    import sys
    import my_cleaner
        
    word_count = {}
    for line in sys.stdin:
# Remove punctuation change to lowercase, and split on whitespace
        word_list = my_cleaner.clean_text(line).split()
# Add words to dictionary
        for word in word_list:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

# Print word counts
    for word in word_count:
        print(word+"\t", word_count[word])
