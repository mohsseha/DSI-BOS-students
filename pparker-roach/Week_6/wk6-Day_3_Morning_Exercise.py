
# coding: utf-8

# Given a tree identical to the format last night write a binary search function that will return either a true or false depending on if the number is in the tree. I.e.:
# Def findItem(tree,int)
# Will return a Boolean that returns true only when int is part of tree.
# The tree is in the format of yesterday's morning exercise but will be sorted. I.e. Left sub-tree is always smaller than "node" and right subtree is always bigger than "node".
# 
# Here is an example of a sorted tree:
# 
# ```tree = {"node": 4, "left": {"node": 1}, "right": {"node": 7, "left": {"node": 5},}}```

# In[ ]:

new_tree = {"node": 4, "left": {"node": 1}, "right": {"node": 7, "left": {"node": 5},}}
if 7 in new_tree.values():
    print{'asdf'}
else:
    print("asdfasasfdasd")


# In[ ]:

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
    


# In[ ]:

print(finditem(new_tree, 5))


# In[ ]:



