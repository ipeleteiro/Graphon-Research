import numpy as np
import math

A = 0
A = np.array([[1,2,3],[4,5,1],[1,2,1],[4,1,5]])

print(A[:,1:4])
print(A[:,1])

D = np.array([1,2,3])
print(D)

# np.array
# np.transpose (and the equivalent method a.T)
# np.ndarray.shape (A.shape)
# np.dot (and the equivalent method a.dot(b) )          Note that in Python, np.dot(a, b) is the matrix product a@b, not the dot product 
# np.sign
# np.sum (look at the axis and keepdims arguments)
# Elementwise operators +, -, *, /

# A[:,1]  --> elements at index 1 for each row
# A[:,1:2] --> vector matrix of elements at index 1

# returns row vector out of list
def rv(value_list):
    return np.array([value_list])
# retusn column vector out of list
def cv(value_list):
    return np.transpose(rv(value_list))

def length(col_v):
    return math.sqrt(np.sum(col_v**2))

def normalize(col_v):
    return col_v/length(col_v)

# takes a 2D array and returns the final column as a two dimensional array
def index_final_col(A):
    return A[:,-1:]

print(index_final_col(A).tolist())


data = np.array([[150,5.8],[130,5.5],[120,5.3]])
def transform(data):
    return np.transpose([np.dot(data, np.array([1, 1]))])
print(transform(data))

print(A.shape)