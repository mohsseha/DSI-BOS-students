
# coding: utf-8

# # Gradients
# 
# The [gradient of a function](https://en.wikipedia.org/wiki/Gradient) is a multivariate derivate with a crucial property -- the gradient points in the direction of the greatest rate of increase of the function. Many physical processes can be modeled by gradient and gradient flows, such as the flow of water down a mountain and the movement of charged particles in electromagnetic potentials.
# 
# As data scientists we use gradient descent to maximize or minimize various functions. For example, to find a good model fit we could attempt to minimize a loss function by following the gradient through many iterations in parameter-space. In particular, gradient descent can be used for [linear regression](https://en.wikipedia.org/wiki/Gradient_descent#Solution_of_a_linear_system). Let's take a close look.
# 
# If we want to minimize a multivariate function $f(\mathbf{a})$ -- typically a function of our parameters $\mathbf{a} = (a_1, \ldots, a_n)$ computed on our dataset -- we start with a guess $\mathbf{a}_1$ and compute the next step using the gradient, denoted by $\nabla f$:
# 
# $$ \mathbf{a}_2 = \mathbf{a}_1 - \lambda \nabla f(\mathbf{a}_1)$$
# 
# Note the differences in notation carefully -- bold face indicates a vector of parameters. The variable $\lambda$ is a parameter that controls the step size and is sometimes called the _learning rate_. Essentially we are taking a local linear approximation to our function, stepping a small bit in the direction of greatest change, and computing a new linear approximation to the function. We repeat the process until we converge to a minimum:
# 
# $$ \mathbf{a}_{n+1} = \mathbf{a}_n - \lambda \nabla f(\mathbf{a}_n)$$
# 
# This is the _gradient descent_ algorithm. It is used for a variety of machine learning models including some that you will learn about soon, such as logistic regression, support vector machines, and neural networks.
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/7/79/Gradient_descent.png)
# 

# In[7]:

get_ipython().magic('matplotlib inline')
import random

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import datasets, linear_model


# ## Some helper functions
# The following functions will generate data and polynomial values.

# In[157]:

def function_to_optimize(x, y):
    return 0.5*(x ** 2 + y ** 2)

def gradient(x, y):
    f = function_to_optimize(x, y)
    return (x, y)

def gradient_descent(gradient_func, x0, y0, l=0.1):
    vector = np.array([x0, y0])
    g = gradient_func(x0, y0)
    return vector - l * np.array(gradient(x0, y0))


# In[158]:

import numpy as np
import matplotlib.pyplot as plt

delta = 0.025

x = y = np.arange(-3.0, 3.01, delta)
X, Y = np.meshgrid(x, y)

Z = function_to_optimize(X, Y)

CS = plt.contourf(X, Y, Z)
plt.colorbar()
plt.show()


# In[159]:

delta = 0.025
x = y = np.arange(-3.0, 3.01, delta)
X, Y = np.meshgrid(x, y)
Z = function_to_optimize(X, Y)
CS = plt.contourf(X, Y, Z)

xs = [2]
ys = [2.1]
for i in range(15):
    x, y = gradient_descent(gradient, xs[-1], ys[-1])
    xs.append(x)
    ys.append(y)

plt.scatter(xs, ys, color="black")
plt.plot(xs, ys, color="black")
ax = plt.gca()
ax.arrow(xs[-2], ys[-2], xs[-1] - xs[-2], ys[-1] - ys[-2], head_width=0.5, head_length=0.3, fc='k', ec='k')
plt.show()


# In[160]:

delta = 0.025
x = y = np.arange(-3.0, 3.01, delta)
X, Y = np.meshgrid(x, y)
Z = function_to_optimize(X, Y)
CS = plt.contourf(X, Y, Z)

for (i, j) in [(-1, 2.1), (1, 2.1), (2, 0.4), (-2, -0.4), (-1, -2.1), (1, -2.1)]:
    xs = [i]
    ys = [j]
    for i in range(15):
        x, y = gradient_descent(gradient, xs[-1], ys[-1])
        xs.append(x)
        ys.append(y)
    
    ax = plt.scatter(xs, ys, color="black")
    ax = plt.gca()
    ax.arrow(xs[1], ys[1], xs[3] - xs[1], ys[3] - ys[1], head_width=0.5, head_length=0.3, fc='k', ec='k')
    plt.plot(xs, ys, color="black")

plt.show()


# ## Independent Practice
# 
# Here is a [nice example](http://math.stackexchange.com/questions/770622/gradient-descent-algorithm-always-converges-to-the-closest-local-optima) of when gradient descent fails. Let's implement these functions and see that gradient descent gets stuck.
# 
# The function is:
# $$f(x, y) = \begin{cases}
# 2 x^2 & \quad \text{if $x \leq 1$}\\
# 2  & \quad \text{else}
# \end{cases}$$
# 
# Walk throught the following code samples.

# In[163]:

def func(x):
    if x <= 1:
        return 2 * x * x
    return 2

def gradient(x):
    if x <= 1:
        return 4 * x
    return 0

def gradient_descent(gradient_func, x, l=0.1):
    vector = np.array(x)
    g = gradient_func(x)
    return vector - l * np.array(gradient(x))


def iterate(gradient, x0, n=10):
    xs = [x0]
    ys = [func(x0)]
    for i in range(n):
        x = gradient_descent(gradient, xs[-1], l=0.1)
        xs.append(x)
        ys.append(func(x))
    return xs, ys


# In[172]:

xs = np.arange(-2, 3, 0.1)
ys = map(func, xs)

plt.plot(xs, ys)

# Start gradient descent at x = -1.5
xs2, ys2 = iterate(gradient, -1.5, n=10)
plt.scatter(xs2, ys2, c='r', s=50)

# Start gradient descent at x = 2
xs2, ys2 = iterate(gradient, 2, n=10)
plt.scatter(xs2, ys2, c='y', s=100)


# Gradient descent works on the left half of the function but not on the right portion because the derivative is flat. Our starting point of $x=2$ never moves and doesn't converge to the center.

# ### Exercise
# 
# Similarly, use the function $f(x) = x^4 - 2* x^2 + x +1$ and apply gradient descent. If you need help with the derivative, you can use [Wolfram Alpha](http://www.wolframalpha.com/calculators/derivative-calculator/).
# 
# Steps:
# * Plot the function and identify the two minima
# * Compute the derivative
# * Using gradient descent, find two starting points that converge to different minima
# 
# Questions:
# * What does this tell you about the end result of gradient descent?
# * What are the implications for putting gradient descent into practice?

# ### Bonus Exercise
# 
# Use gradient descent to find the minimum of the function
# $$f(x, y) = - e^{-x^2 - 4y^2}$$

# In[ ]:



