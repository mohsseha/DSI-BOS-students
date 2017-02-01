
# coding: utf-8

# Write two functions to convert a Roman Numeral string to an integer and conversly, a Roman Numeral Integer to a string
# 
# <img src="roman.png" width="500">
# 
# When one or more numeral is used to form a number, the value of each symbol is (generally) added together from left to right.
# 
# II = 2
# * XXX (10+10+10) = 30
# * LII (50+1+1) = 52
# * MMLVII (1,000+1,000+50+5+1+1) = 2,057
# 
# In some instances, a lower numeral placed in front of a larger numeral indicates that the lower numeral should be subtracted from the larger.
# 
# * 29 = XXIX (10+10+(10-1))
# * 399 = CCCXCIX (100+100+100+(100-10)+(10-1))
# * 444 = CDXLIV ((500-100)+(50-10)+(5-1))

# In[110]:


def str_to_int(rn):
    # adding Roman Numeral symbols and thier values to this dictionary will expand the range of input acceptable by
    # this routine
    romNums = { 'M':  1000,
                'CM': 900,
                'D':  500,
                'CD': 400,
                'C':  100,
                'XC': 90,
                'L':  50,
                'XL': 40,
                'X':  10,
                'IX': 9,
                'V':  5,
                'IV': 4,
                'I':  1}

    num = 0
    rn = rn.upper()
    if len(rn) >= 2:
            if rn[0:2] in romNums.keys():
                num = num + int(romNums[rn[0:2]])
                return num + str_to_int(rn[2:])
            elif rn[0] in romNums.keys():
                num = num + int(romNums[rn[0]])
                return num + str_to_int(rn[1:])
    elif len(rn) == 1:
        num = num + int(romNums[rn[0]])
        return num
    else:
        return num
    return num


# In[108]:

def int_to_str(num):
    rn = ""
    if (num > 1000):
        return "Your number must be less than 1000"
    if num >= 1000:
        rn = rn + "M"
        return rn + str(int_to_str(num - 1000))
    elif 1000 > num >= 900:
        rn = rn + "CM"
        return rn + str(int_to_str(num - 900))
    elif 900 > num >= 500:
        rn = rn + "D"
        return rn + str(int_to_str(num - 500))
    elif 500 > num >= 400:
        rn = rn + "CD"
        return rn + str(int_to_str(num - 400))        
    elif 400 > num >= 100:
        rn = rn + "C"
        return rn + str(int_to_str(num - 100))
    elif 100 > num >= 90:
        rn = rn + "XC"
        return rn + str(int_to_str(num - 90))        
    elif 90 > num >= 50:
        rn = rn + "L"
        return rn + str(int_to_str(num - 50))
    elif 50 > num >= 40:
        rn = rn + "XL"
        return rn + str(int_to_str(num - 40))        
    elif 40 > num >= 10:
        rn = rn + "X"
        return rn + str(int_to_str(num - 10))
    elif 10 > num >= 9:
        rn = rn + "IX"
        return rn + str(int_to_str(num - 9))        
    elif 9 > num >= 5:
        rn = rn + "V"
        return rn + str(int_to_str(num - 5))
    elif 5 > num >= 4:
        rn = rn + "IV"
        return rn + str(int_to_str(num - 4))        
    elif 4 > num >= 1:
        rn = rn + "I"
        return rn + str(int_to_str(num - 1))        

    return rn


# In[109]:

for i in range(1, 1000):
    roman_numeral = int_to_str(i)
    numeral = str_to_int(roman_numeral)
    if numeral != i:
        print("broken at ", i)


# In[ ]:




# In[ ]:



