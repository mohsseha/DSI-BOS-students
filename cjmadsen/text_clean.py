def clean_text(text):
    symbols = ['`', '"', "'", ';', ':', '?', "!", '.', ',','-',"_",'/', "(", ")", "[", "]", "{", '}',"\n" ]
    for item in symbols:
            text = text.replace(item, '') #remove above symbols from text
    text = text.lower() #convert the string to lowercase
    text = text.split(' ') #split the string into a list on spaces
    return text
