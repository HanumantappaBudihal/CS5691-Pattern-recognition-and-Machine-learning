import imp
import numpy as np
import math


def fit(data):
    """
    Find out the variance matrix for the given data

    Parameters
    ----------
    data :  matrix data 
        
    Returns
    -------        
    eigen_values and eigen_vectors

    """
    covariance = data.T @ data  # Matric multlication
    eigen_values, eigen_vectors = np.linalg.eigh(covariance)

    return eigen_values, eigen_vectors


def kernalPCA(data, function_pointer, d_value, no_ocmponents):
    """
    kernal PCA

    Parameters
    ----------
    data 
    function_pointer
    d_value
    no_ocmponents
    
    Returns
    -------        
    eigen_values and eigen_vectors

        """
    nk = []
    for i in range(len(data)):
        n = []
        for j in range(len(data)):
            n.append(function_pointer(data[i, :], data[j, :], d_value).item())
        nk.append(n)

    # nk= K @ K.T
    d = np.matrix(nk)

    eigen_values, eigen_vectors = np.linalg.eigh(d)
    eigen_values = eigen_values[-1:(-1+-1*no_ocmponents):-1]
    eigen_vectors = eigen_vectors[:, -1:(-1)+no_ocmponents:-1]

    # this will put the eigenvector in row
    eigen_vectors = [(eigen_vectors[:, i]/(len(data)*eigen_values[i]) ** (1/2)).getA1() for i in range(no_ocmponents)]
    eigen_vectors = np.matrix(eigen_vectors)

    projected_data = d  @ eigen_vectors.T

    return eigen_values, projected_data

 
def polynominal_function(x, y, v):
    """
    Implementation for this (1 + x^Ty)^d for d = {2,3} function

        """
    return (1 + x @ y.T)**v


def gauusian_function(x, y, p):
    """
    Implementation for this e^(-(x-y)(x-y)^T / 2*(sigam)^2) for d = {2,3} function

        """
    return np.matrix(math.exp(((y-x) @ (x-y).T)/(2*p*p)))
