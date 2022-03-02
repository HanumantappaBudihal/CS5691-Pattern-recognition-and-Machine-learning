# ***************************************************************************************************************************
# File      : Common.py 
# Purpose   : Provides access to common utility functionalities ( read file)
# Author    : Hanumantappa Budihal
#
# Date      : 23-02-2022 
# Bugs      : NA
# Change Log:
# ****************************************************************************************************************************/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    """
	Read the data from the file using pandas framework and
    return the data.

	Parameters
	----------
	file_path : 
              path for file 
	Returns
	-------        
    return the data from the file
     
	"""
    data = pd.read_csv(file_path, header=None)
    data = np.matrix(data)

    return data


def center_data(data):
    """
	Centred the data by subtracing the mean from each data point.

	Parameters
	----------
	data : 
          input data 
	Returns
	-------        
    centred data
     
	"""
    data = data-data.sum(axis=0)/len(data)

    return data


def plot_data(original_data, projected_data, eigenvalues, colour):
    """
	Plot the given set of the data sets

	Parameters
	----------
	data : 
          original_data : original data set
          projected_data : projected data set using pca
          eigenvalues : eigen values
          colour : data points color
	Returns
	-------        
    None :

    Plot the all 3 type graph from given data     
	"""

    # original data
    plt.figure(figsize=(18, 5))
    
    plt.subplot(131)
    plt.title("original data")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.plot(original_data[:, 0], original_data[:, 1], colour)

    # projected data
    plt.subplot(132)
    plt.title("projected data")
    plt.xlabel("x-axis/first major eigen vector")
    plt.ylabel("y-axis/second major eigen vector")
    plt.plot(projected_data[:, 0], projected_data[:, 1], colour)

    # eigenvlaues plot
    plt.subplot(133)
    plt.title("eigenvlaues plot")
    plt.xlabel("x-axis/no of eigen values")
    plt.ylabel("y-axis/eigen vlaues")
    plt.plot([i for i in range(1, len(eigenvalues)+1)], eigenvalues, colour)


def random(dataset, k):
    """
    Create random cluster centroids.

    Parameters
    ----------
    dataset : numpy array
        The dataset to be used for centroid initialization.
    k : int
        The desired number of clusters for which centroids are required.
    Returns
    -------
    centroids : numpy array
        Collection of k centroids as a numpy array.
    """

    centroids = []
    m = np.shape(dataset)[0]

    for _ in range(k):
        random_number = np.random.randint(0, m-1)
        centroids.append(np.ravel(dataset.iloc[random_number, :]))

    return np.array(centroids)


def add(dataset, k, random_state=42):
    """
    Create random cluster centroids.

    Parameters
    ----------
    dataset : numpy array
        The dataset to be used for centroid initialization.
    k : int
        The desired number of clusters for which centroids are required.
    Returns
    -------
    centroids : numpy array
        Collection of k centroids as a numpy array.
    """
    np.random.seed(random_state)
    m = np.shape(dataset)[0]

    centroids = []

    for _ in range(k):
        random_number = np.random.randint(0, m-1)
        centroids.append(np.ravel(dataset.iloc[random_number, :]))

    return np.array(centroids)
