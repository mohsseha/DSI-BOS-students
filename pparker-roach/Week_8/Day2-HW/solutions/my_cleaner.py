def clean_text(line):
    words = []
    keep = "abcdefghijklmnopqrstuvwxyz0123456789" # not the smartest way of doing this but good enough :) 

    for word in line.split(" "):
        word=word.lower()
        word = ''.join(ch for ch in word if ch in keep)
        words.append(word)

    return words
