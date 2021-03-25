import numpy as np
from numba import njit
import itertools

def linear_kernel(X, Y):
    """
    Args: 
    X: np.array, shape (batch_X, ...)
    Y: np.array, shape (batch_Y , ...)
    
    Return: np.array, Kernel, shape(batch_X, batch_Y)
    """ 
    X_flat = X.reshape((X.shape[0], -1))
    Y_flat = Y.reshape((Y.shape[0], -1))
    return X_flat@Y_flat.T

def gaussian_kernel(X, Y, sigma = 1, typ = 'matrix'):
    """
    Args: 
    X: np.array, shape (batch_X, ...)
    Y: np.array, shape (batch_Y , ...)
    sigma : std
    
    Return: np.array, Kernel, shape(batch_X, batch_Y)
    """ 
    diff = X[:, None] - Y[None]
    if typ == 'matrix':
        diff = (diff**2).sum(axis = (2, 3))
    else:
        diff = (diff**2).sum(axis = 2)
    return np.exp(-diff/(2*sigma**2))

@njit
def similarity(lst, X, Y, length, mismatch, λ, norm = True):
    l = X.shape[0]
    k = X.shape[1]
    n = Y.shape[0]
    z = lst.shape[0]
    simx = np.zeros((l, z))
    simy = np.zeros((n, z))

    # Compute representation of X and Y
    for j in range(z):
        sample = lst[j]                                    
        for m in range(k - length + 1):
            for h in range(l):
                sample2 = X[h, m:(m+length)]
                mis = (np.abs(sample - sample2)).sum()/2
                if mis <= mismatch:
                    simx[h, j] += λ**mis

            for h in range(n):
                sample2 = Y[h, m:(m+length)]
                mis = (np.abs(sample - sample2)).sum()/2
                if mis <= mismatch:
                    simy[h, j] += λ**mis

    if norm:
        normx = np.sqrt((simx**2).sum(axis = 1)).reshape(-1, 1)
        normy = np.sqrt((simy**2).sum(axis = 1)).reshape(-1, 1)
        K = simx@simy.T/(normx@normy.T)
    else:
        K = simx@simy.T      
    
    return K

def mismatch_kernel(X, Y, length = 6, mismatch = 1, λ = 1, norm = True):
    """
    Args: 
    X: np.array, shape (batch_X, 101, 4)
    Y: np.array, shape (batch_Y , 101, 4)
    
    Return: np.array, Kernel, shape(batch_X, batch_Y)
    """ 
    lst = np.array(list(itertools.product(np.eye(X.shape[2]), repeat = length)))
    K = similarity(lst, X, Y, length, mismatch, λ, norm)
            
    return K

def SumKernels(X, Y, kernels):
    n, l = X.shape[0], Y.shape[0]
    K = np.zeros((n, l))
    for kernel in kernels:
        K += kernel(X, Y)
    
    return K/len(kernels)
