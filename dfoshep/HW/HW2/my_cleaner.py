def clean_text(line):
    import re
    words = []

    for word in line.split(" "):
        word=word.lower()
        word = re.sub(r'[^a-z0-9 \n]', '', word)

        words.append(word)

    return words
