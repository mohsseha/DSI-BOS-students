def clean_text(x):
    import re
    x_clean = re.sub(r"[!, ?, ., ;, \,, \", 0-9, -]", " ", x)
    x_clean_lower = x_clean.lower()
    return(x_clean_lower)
