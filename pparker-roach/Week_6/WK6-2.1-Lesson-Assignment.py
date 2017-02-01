
# coding: utf-8

# In[120]:

def sum_nodes(tree):
    #This function recursively traverses a binary tree and returns the sum of the values of the nodes
    total = 0
    if "left" in tree.keys():
        total = total + sum_nodes(tree["left"])
    if "right" in tree.keys():
        total = total + sum_nodes(tree['right'])
    return total + tree["node"]


# In[ ]:




# In[121]:

a_tree = {"node": 4, "left": {"node": 4}, "right": {"node": 3, "left": {"node": 5},}}


# In[122]:

a_tree["right"]


# In[123]:

print(sum_nodes(a_tree))


# In[ ]:




# In[ ]:



