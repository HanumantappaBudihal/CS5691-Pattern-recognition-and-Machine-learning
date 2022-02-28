import pandas as pd
from utilities.KMeans import KMeans

import matplotlib.pyplot as plt
import utilities.common as common
import copy

if __name__ == '__main__':

# ############################ Questsion 2 -> (i) Part #######################################

    print("---------------------------- Q2 - part (i) Output------------------------")

    # Path of the dataset file
    dataset_file = 'dataset/dataset.csv'
    # load datasets from files
    dataset = pd.read_csv(dataset_file, sep=',', header=None)

    # # k = 4 and 5-different random initilization    
    # kmeans1 = KMeans(n_cluster=4)
    # for i in range(1, 6):
    #     clone= copy.deepcopy(dataset)
        
    #     kmeans1.fit(dataset)
    #     kmeans1.save_figures('plots/Q2', 4, 'Q1a{}'.format(i))


# ############################ Questsion 2 -> (i) Part #######################################

    print("---------------------------- Q2 - part (ii) Output------------------------")
    #  fixed initialization and K = {2,3,4,5}

    #initialization centroid for cluster
    centroids=[]
    centroidk2=[[0, 0],[-5.0 , -5.0]]    
    
    for i in range(2, 3):
        kmeans1 = KMeans(n_cluster=i)
        clone= copy.deepcopy(dataset)
        
        kmeans1.fit(dataset,centroidk2)
        kmeans1.save_figures('plots/Q2', 4, 'Q2b{}'.format(i))


    print("---------------------------- Processing completed ------------------------")
