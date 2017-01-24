import re

# Define word count function
def count_words(input):
    "Count number of words in input text"
# Remove numbers and relevant punctuation
#    input = str(input)
    words_only = re.sub(r"[!, ?, ., ;, \,, \", 0-9]", " ", input)
# Convert to lower case, and split on whitespace
    word_list = words_only.lower().split()
# Initialize and fill word count dictionary
    word_count = {}
    for word in word_list:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
# Print word counts
    print("NOTE: Output is also saved to output.txt")
    with open('output.txt', 'w') as fout:
        for word in word_count:
            print(word+"\t"+str(word_count[word]))
            fout.write(word+"\t"+str(word_count[word]))
            fout.write("\n")

# Call function from command line
if __name__ == "__main__":
    import sys
    count_words(sys.stdin.read())
