
# coding: utf-8

# In[ ]:




# In[58]:

l = [3,6,7,11,15,84]


# In[24]:

def build_tree(l):
    tree = {}
    l.sort()
    if len(l) == 0:
        return
    if len(l)%2 == 0:
        tree["node"] = l[int(len(l)/2)]
        left_list = l[:int(len(l)/2)]
    if len(l)%2 != 0:
        tree["node"] = l[int(len(l)/2)-1]
        right_list = l[:int(len(l)/2)-1]
    tree['left'] = build_tree(left_list)
    tree['right'] = build_tree(right_list)
    return(tree)


# In[25]:

build_tree(l)


# In[59]:

def build_tree(l):
    

    return(tree)


# In[60]:

l = [3,6,7,11,15,84]


# In[61]:

build_tree(l)


# In[62]:

import numpy as np


# In[63]:

test_list = [-3,-2,-1,0,1,2,3]

def build_tree(number_list):
    sorted_list = np.sort(number_list)
   
    median_value_index = int(len(sorted_list) / 2)
    median = sorted_list[median_value_index]
   
    dictionary = {}
   
    left_dictionary = {'node': sorted_list[0]}
    for index in range(1, median_value_index + 1):
        if index < median_value_index:
            left_dictionary = {'node': sorted_list[index], 'left': left_dictionary}
   
    right_dictionary = {'node': sorted_list[len(sorted_list) - 1]}
    for index in reversed(range(median_value_index + 1, len(sorted_list - 1))):
        if index > median_value_index:
            right_dictionary = {'node': sorted_list[index - 1], 'right': right_dictionary}
   
    dictionary = {'left': left_dictionary, 'node': right_dictionary['node'], 'right': right_dictionary['right']}

   
    print("Median: ", median)
    print(sorted_list)
   
    return dictionary



# In[67]:

test_tree = build_tree(test_list)
test_tree


# In[65]:

def finditem(tree,int):
    #given a binary tree and and integer, return a true if the integer is in the tree, else false
    found = False
    if tree["node"] == int:
        return True
    elif ("left" in tree.keys()) and tree["node"] > int:
        #print("going left")
        return finditem(tree["left"], int)
    elif ("right" in tree.keys()) and tree["node"] < int:
        #print("going right")
        return finditem(tree["right"], int)
    else:
        return found
    


# In[66]:

for i in test_list:
    print(finditem(test_tree, i))
    


# In[96]:

import sys
sys.getrecursionlimit()


# In[163]:

def build(list, tree):
    #given a binary tree and and integer, return a true if the integer is in the tree, else false
    if len(list) == 0: return
    
    if (bool(tree) == False):
        tree['node'] = list[0]
        print("make first node = ", tree['node'])
        list.pop(0)
        print(list)
        return build(list, tree)
    
    if (tree["node"] == list[0]):
        print("first node already here")
        list.pop(0)

     

        
    if ("right" in tree.keys()) and tree["node"] < list[0]:
        print("going right")
        tree['right'] = build(list, tree)
        return
    elif tree['node'] < list[0]: 
        print("create right node ", tree)
        tree['node'] = list[0]
        return build(list.pop(0), tree)
        
        
#     elif ("left" in tree.keys()) and tree["node"] > list[0]:
#         print("going left")
#         tree['left'] = build(list, tree)
#         return tree['left']
#     elif tree['node'] > list[0]:
#         print("create left node ", tree)
#         tree['left'] = build(list, tree)
#         return tree['left']

#     elif tree['node'] < list[0]:
        
#         print("create right node ", tree)
#         tree['right'] = build(list, tree)
#         return tree['right']

    


# In[ ]:




# In[164]:

l = [3,3,6,7,11,15,84]
some_tree ={}
build(l, some_tree)
some_tree


# In[166]:

some_tree


# In[177]:

new_tree = {"node": 4, "left": {"node": 1}, "right": {"node": 7, "left": {"node": 5},}}
if "node" in new_tree.values():
    print("true")
else:
    print("asdfasasfdasd")


# In[178]:

max_int=np.max(l)


# In[179]:

max_int


# In[ ]:



