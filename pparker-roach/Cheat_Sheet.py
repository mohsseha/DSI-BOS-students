
# coding: utf-8

# # Cheat Sheet
# ## Dataframe containing useful information regarding coding tricks that I find useful
# ### structure = description of what it does : function name

# In[1]:

import pandas as pd


# In[2]:

cheat_sheet = pd.read_csv("cheat_sheet.csv")


# #### here i how to add a new row to the dataframe
# 
# cheat_sheet = cheat_sheet.append({"Language/Package":"Name of the language or package",
#     "Functionality":"some text",
#     "Function":"function name",
#     "Example":"optional example of the functionality"},
#     ignore_index=True)

# In[3]:

#cheat_sheet.append({"Language/Package":"Pandas","Functionality":"add a new row to a dataframe", "Function":"df.append()", "Example":"cheat_sheet.append({'Language/Package':'Name of the language or package','Functionality':'some text', 'Function':'function name', 'Example':'optional example of the functionality'}, ignore_index=True)"}, ignore_index=True)


# cheat_sheet = cheat_sheet.append({"Language/Package":"Pandas",
#     "Functionality":"reading in a csv file with wrong encoding for Win 10",
#     "Function":"encoding parameter for pd.read_CSV",
#     "Example":"encoding = 'iso-8859-1' or encoding = 'cp1252' or encoding = 'latin1'"},
#     ignore_index=True)

# ### I am going to create a function to ask for data from the user and create a new row in my cheat sheet

# In[ ]:

def add_new_insight(my_sheet):
    print("Enter your new insight below...")
    language = input("\tWhat language or package is this input related to?: ")
    functionality = input("\tWhat functionality does this insight provide: ")
    function = input("\tWhat function is this insight related to?: ")
    example = input("\tGive an example of this insight. ")
    print("Here is what I got from you...")
    print("\tLanguage/Package = {}\n\tFunctionality = {}\n\tThe Function is = {}\n\tYour Example is{}".format(language, functionality, function, example ))
    commit = input("Would you like me to commit this to your Cheat Sheet? (y/n)")
    if (commit == 'y') or (commit == 'Y'):
        my_sheet = my_sheet.append({"Language/Package":language,
        "Functionality":functionality,
        "Function":function,
        "Example":example},
        ignore_index = True) 
        return my_sheet
    else: return my_sheet
    
cheat_sheet = add_new_insight(cheat_sheet)
cheat_sheet
#cheat_sheet = cheat_sheet.append({"Language/Package":"Pandas",
#    "Functionality":"reading in a csv file with wrong encoding for Win 10",
#    "Function":"encoding parameter for pd.read_CSV",
#    "Example":"encoding = 'iso-8859-1' or encoding = 'cp1252' or encoding = 'latin1'"},
#    ignore_index=True)


# In[21]:

cheat_sheet.to_csv('cheat_sheet.csv', index=False)


# In[ ]:



