import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
import pandas as pd
import copy
import numpy

plt.style.use('ggplot')
np.random.seed(1)

def dist(a, b, ax=1):
    """ Euclidean distance """
    return np.linalg.norm(a-b, axis=ax)

def kmeans(X, k):
    C_x = np.random.uniform(0, np.max(X[0]), size=k)
    C_y = np.random.uniform(0, np.max(X[1]), size=k)
    C = np.array(list(zip(C_x, C_y)), dtype=np.float32)

    C_old = np.zeros(C.shape)
    clusters = np.zeros(X.shape[1])
    error = dist(C, C_old, None)
    
    flag = True
    
    while error != 0:
        for i in range(X.shape[1]):
            distances = dist((X[0][i], X[1][i]), C)
            if numpy.isnan(distances).any() and flag:
                flag = False
                print(distances)
                print("Coords: ")
                print(X[0][i], X[1][i])
                print("C: ")
                print(C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        C_old = copy.deepcopy(C)
        for i in range(k):
            points = np.array([(X[0][j], X[1][j]) for j in range(X.shape[1]) if clusters[j] == i])
            if len(points) == 0:
                C[i] = 0
            else:
                C[i] = np.mean(points, axis=0)
        error = dist(C, C_old, None)
    
    return clusters
def similarity(a, b):
    return np.exp(-(abs(a-b)**2) / 10)


import scipy.sparse.linalg
def spectral_clustering(affinity, k):
    def laplacian(A):
        D = np.zeros(A.shape)
        w = np.sum(A, axis=0)
        D.flat[::len(w) + 1] = w ** (-0.5)
        return D.dot(A).dot(D)
    L = laplacian(affinity)
    _, eigvec = scipy.sparse.linalg.eigs(L, k)
    X = eigvec.real
    rows_norm = np.linalg.norm(X, axis=1, ord=2)
    labels = kmeans((X.T / rows_norm), k)
    return labels

def compute_affinity(X):
    def squared_exponential(x, y, sig=0.8, sig2=1):
        """ Models smooth functions
        
        Function from previous spectral clustering project from tut """
        norm = numpy.linalg.norm(x - y)
        dist = norm * norm
        return numpy.exp(- dist / (2 * sig * sig2))
    
    N = X.shape[0]
    res = np.zeros((N, N))
    sig = []
    for i in range(N):
        dists = []
        for j in range(N):
            dists.append(np.linalg.norm(X[i] - X[j]))
        dists.sort()
        sig.append(np.mean(dists[:5]))

    for i in range(N):
        for j in range(N):
            res[i][j] = squared_exponential(X[i], X[j], sig[i], sig[j])
    return res

def plot_clusters(X, clusters, k, title="Title here"):
    symbols = ['x', 'o', '^', '.', '1', '2', '3', '4']
    fig, ax = plt.subplots()
    for i in range(k):
        points = np.array([(X[0][j], X[1][j]) for j in range(X.shape[1]) if clusters[j] == i])
        if len(points) != 0:
            ax.scatter(points[:, 0], points[:, 1], s=40, marker=symbols[i])
    plt.title(title)
    plt.savefig("test.png", dpi=500)