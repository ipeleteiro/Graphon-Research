import numpy as np
import math

# np.array
# np.transpose (and the equivalent method a.T)
# np.ndarray.shape
# np.dot (and the equivalent method a.dot(b) )          Note that in Python, np.dot(a, b) is the matrix product a@b, not the dot product 
# np.sign
# np.sum (look at the axis and keepdims arguments)
# Elementwise operators +, -, *, /

def rv(value_list):
    return np.array([value_list])
# retusn column vector out of list
def cv(value_list):
    return np.transpose(rv(value_list))
def length(col_v):
    return math.sqrt(np.sum(col_v**2))

# signed perpendicular distance from the hyperplane specified by theta, theta_0 to this point x
def signed_dist(x, th, th0):
    return (np.transpose(th).dot(x)+th0)/length(th)

# returns
        # +1 if x is on the positive side of the hyperplane encoded by (th, th0)
        # 0 if on the hyperplane
        # -1 otherwise.
def positive(x, th, th0):
    dist = np.sign(np.transpose(th).dot(x)+th0)/length(th)      # don't need to divide by th, the sign will be the same
    if dist > 0:
        return np.array([[1]])
    elif dist == 0:
        return np.array([[0]])
    else:
        return np.array([[-1]])

# 1 by 5 array of boolean values, either True or False, 
# indicating for each point in data and corresponding label in labels whether it is correctly classified by hyperplane
# A = labels == positive(data, th, th0)

# returns the number of points for which the label is equal to the output of the positive function on the point.
def score(data, labels, th, th0):
    return np.sum(labels == positive(data, th, th0))

A = np.array([[1,1,1],[2,2,2],[3,3,3]])
B = np.array([[1,2,3]])
print(np.sum(A==B, axis=1))

def score_mat(data, labels, ths, th0s):
   pos = np.sign(np.dot(np.transpose(ths), data) + np.transpose(th0s))      # don't need to divide by th, the sign will be the same
   return np.sum(pos == labels, axis = 1, keepdims = True)

def best_separator(data, labels, ths, th0s):
   best_index = np.argmax(score_mat(data, labels, ths, th0s))
   return cv(ths[:,best_index]), th0s[:,best_index:best_index+1]