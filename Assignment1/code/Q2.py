import pandas as pd
import numpy as np

from utilities.KMeans import KMeans
import utilities.Common as common
import utilities.SpectralClustering as sc
import copy

if __name__ == '__main__':

    # ############################ Questsion 2 -> (i) Part #######################################

    print("---------------------------- Q2 - part (i) Output------------------------")

    # Path of the dataset file
    dataset_file = 'dataset/dataset.csv'
    # load datasets from files
    dataset = pd.read_csv(dataset_file, sep=',', header=None)

    # k = 4 and 5-different random initilization centroids
    number_of_cluster = 4
    kmeans1 = KMeans(n_cluster=number_of_cluster)

    for i in range(1, 6):
        # Get the random centroid points everytime
        initial_centroids = common.random(dataset, number_of_cluster)
        clone = copy.deepcopy(dataset)

        kmeans1.fit(dataset,initial_centroids)
        kmeans1.save_figures('plots/Q2', number_of_cluster, 'Q2a{}'.format(i))

# ############################ Questsion 2 -> (ii) Part #######################################

    print("---------------------------- Q2 - part (ii) Output------------------------")
    
    #fixed initialization and K = {2,3,4,5}

    #initialization centroid for cluster
    initial_centroids = common.random(dataset, 2)

    for i in range(2, 6):

        kmeans1 = KMeans(n_cluster=i)
        clone = copy.deepcopy(dataset)
        initial_centroids = common.add(clone, i)

        kmeans1.fit(dataset,initial_centroids)
        kmeans1.save_figures('plots/Q2', 4, 'Q2b{}'.format(i))

# ############################ Questsion 2 -> (iii) Part #######################################

    print("---------------------------- Q2 - part (iii) Output------------------------")
    
    file = open("dataset/Dataset.csv")
    data = np.loadtxt(file, delimiter=",")
    data = np.transpose(data)
    
    affinity1 = sc.compute_affinity(data.T)
    k1 = sc.spectral_clustering(affinity1, 4)
    sc.plot_clusters(data, k1, 4, "plots/Q2","Spectral on dataset, k=4")

    print("---------------------------- Processing completed ------------------------")
   