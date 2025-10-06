import numpy as np

def row_average_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (m,1) array where each entry is the average of a row
    """
    (m, n) = x.shape
    return np.dot(x, np.ones((n,1)))/n

# simpler
def col_average_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (n,1) array where each entry is the average of a column
    """
    return np.mean(x, axis=0,keepdims=True).T