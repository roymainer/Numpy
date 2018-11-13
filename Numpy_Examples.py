#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Numpy Array


# In[2]:


import numpy as np
import time
import sys

l = range(1000)
print(sys.getsizeof(5)*len(l))

array = np.arange(1000)
print(array.size* array.itemsize)  # numpy is memory effective compared to python list


# In[5]:


SIZE = 1000000

l1 = range(SIZE)
l2 = range(SIZE)

a1 = np.arange(SIZE)
a2 = np.arange(SIZE)

# Python list
start = time.time()
result = [(x+y) for x,y in zip(l1, l2)]
print("Python list took: ", (time.time() - start)*1000)

start = time.time()
result = a1 + a2
print("Numpy list took: ", (time.time() - start)*1000)  # Numpy is also faster


# ## Two dimensions array

# In[12]:


a = np.array([5, 6, 9])  # one dim array
a[0]

a = np.array([[1, 2], [3, 4], [5, 6]])  # one dim array
print("Number of dimensions: ", a.ndim)  # print the number of dimensions
print("Number of elements: ",a.size)  # total number of elements
print("Matrix dimensions: ", a.shape)  # matrix shape


# In[14]:


np.zeros((3,4))  # create a 3x4 zero matrix


# In[16]:


np.ones((3,4))  # create a 3x4 ones matrix


# In[20]:


np.linspace(1, 5, 5)  # linear sequence of numbers


# In[22]:


a.reshape(2,3)  # reshape a matrix/array
b = a.ravel()  # matrix to array


# In[27]:


a.min()
a.min()
print("Sum of all elements: ", a.sum())
a.sum(axis=1)  # the sum of axis 1 elements
np.sqrt(a)  # square root
np.std(a)  # standard deviation


# In[30]:


a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

a+b
a*b
a/b


# ## Slicing

# In[41]:


a = np.array([6, 7, 8])
a[0:2]  # slicing the list
a[-1]

a = np.array([[6,7,8], [1,2,3], [9,3,2]])
a[1,2]
a[-1]  # last column
a[-1, 0:2]  # first two elements from last row
a[:, 1:3]  # all rows, second and third columns


# ## Elements manipulations

# In[52]:


a = np.arange(6).reshape(3,2)
b = np.arange(6, 12).reshape(3, 2)
np.vstack((a,b))
np.hstack((a,b))


a = np.arange(30).reshape(2,15)
result = np.hsplit(a, 3)
result


# In[55]:


a = np.arange(12).reshape(3,4)
b = a > 4  # return a bool matrix of numbers in a bigger than 4
b
a[b]  # extracts all elements > 4 from a
a[b] = -1  # replace all elements in a bigger then 4 with -1


# ## Iterating

# In[58]:


for cell in a.flatten():
    print(cell)


# In[63]:


for x in np.nditer(a, order='F', flags=['external_loop']):  # print in Fortran order (by column, not by row)
    print(x)


# In[66]:


for x in np.nditer(a, order='C', op_flags=['readwrite']):
    x[...]=x*x
a


# In[67]:


for x,y in np.nditer([a,b]):
    print(x,y)

