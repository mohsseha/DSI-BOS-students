
# coding: utf-8

# In[15]:

import pandas as pd
import matplotlib.pyplot as plt

x = [1,3,8,7,9]
y = [1,3,7,8,10]
#x = range(3)
#y = range(3)
plt.plot(x,y)
plt.show()
y


# In[23]:

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import random

def fun(x, y):
  return x**2 + y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(0.0, 3.0, 0.05)
#x = [1,3,8,7,9]
#y = [1,3,7,9,10]
X, Y = np.meshgrid(x, y)
zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
print(np.ravel(X))
ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


# In[29]:

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


# In[39]:

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

fig=plt.figure()
ax = Axes3D(fig)

x=[-2,-1,0,1,2]
y=[-2,-1,0,1,2]
z=[3,3,5,6,7]
x, y= np.meshgrid(x,y)


ax.plot_surface(x,y,z)

plt.show()


# In[45]:

SHAPE = 4
wo=np.linspace(-2,2,SHAPE)

wo=np.outer(wo,np.ones(SHAPE))
w1=wo.T.copy()
print(wo)
print(w1)


# In[48]:

def cost(w0,w1):
    return ((1-(w0+w1*1))**2+(3-(w0+w1*3))**2+(7-(w0+w1*8))**2+(8-(w0+w1*7))**2+(10-(w0+w1*9))**2)
vCost=np.vectorize(cost)


# In[ ]:



