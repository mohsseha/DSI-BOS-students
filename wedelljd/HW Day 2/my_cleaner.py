import re
def clean_text(line):
    line = re.sub(r'[^a-zA-Z0-9 \n]', '', line)
    line = line.lower()
    return(line)
