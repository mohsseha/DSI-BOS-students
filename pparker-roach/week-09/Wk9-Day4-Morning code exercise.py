
# coding: utf-8

# # Creat a class object titled node where
# 

# In[36]:

class Node(object):
    '''
    Part 1:
    v=Node(4)           # creates a node with value 4
    v.val               # returns 4
    v.left              # returns None
    V.right             # also returns None
    
    v.left = Node(3)    # allows you to add a node tot he let of the main node
    v.right = Node(5)   # adds a node to the right
    
    Part #2
    v.add(6)            # will put 6 in an appropriate place in a binary tree under a node
    v.add(4)            # does nothing because there already is a Node(4)
    v.add([3,6,1,88,4]) # will do the right thing
    '''

    def __init__(self, a_number):
        '''
        '''
        self.val = a_number
        self.left = None
        self.right = None
        
    def left(self, a_node=None):
        '''
        '''
        if a_node == None:
            return self.left
        else:
            self.left = a_node
            return self.left    
        
    def right(self, a_node=None):
        '''
        '''
        if a_node == None:
            return self.right
        else:
            self.right = a_node
            return self.right
    
    def add(self, numbers = None ):
        '''
        '''
        if numbers == None:
            return
        if type(numbers) == int:
            if numbers < self.val:
                if self.left == None:
                    self.left = Node(numbers)
                else:
                    self.left.add(numbers)
            elif numbers > self.val:
                if self.right == None:
                    self.right = Node(numbers)
        elif type(numbers) == list:
            self.add(num)
                


# In[30]:

v = Node(4)


# In[31]:

v.left = Node(3)


# In[32]:

v.left.val


# In[25]:

type(v)


# In[33]:

v.add(7)


# In[35]:

v.right.val


# In[37]:

v.add([1,10,5,18,7])


# In[40]:




# In[ ]:



