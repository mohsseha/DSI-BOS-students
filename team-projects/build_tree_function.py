
# coding: utf-8

# In[5]:

import random
import numpy as np


# In[6]:

# DEBUGGING: Make a test list

test_list = [random.randint(-5, 5) for _ in range(10)]


# In[7]:

# Pseudocode:
#     Sort the list
#     Find the median value and its index in the list
#     Build left- and right-hand trees, starting at the edges
#     Combine the left- and right-hand trees

def build_tree(number_list):
    
    # Sort the list
    sorted_list = np.sort(number_list)
    
    # Find the median's value and index
    median_value_index = int(len(sorted_list) / 2)
    median = sorted_list[median_value_index]
    
    # Initialize the dictionaries
    dictionary = {}
    left_dictionary = {'node': sorted_list[0]}
    right_dictionary = {'node': sorted_list[len(sorted_list) - 1]}
    
    # Build the left and right sides
    for index in range(1, median_value_index):
        left_dictionary = {'node': sorted_list[index], 'left': left_dictionary}
        right_dictionary = {'node': sorted_list[len(sorted_list) - index - 1], 'right': right_dictionary}
    
    # Combine the dictionaries
    dictionary = {'left': left_dictionary, 
                  'node': right_dictionary['node'], 
                  'right': right_dictionary['right']}
    
    # DEBUGGING
    print(sorted_list)
    print("node: ", dictionary['node'])
    print("left: ", dictionary['left'])
    print("right: ", dictionary['right'])
    
    return dictionary


# In[8]:

x = build_tree(test_list)


# In[ ]:





# In[ ]:



