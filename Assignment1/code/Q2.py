import pandas as pd
import copy

from utilities.KMeans import KMeans
import utilities.Common as common
import utilities.SpectralClustering as sc

if __name__ == '__main__':

    # ############################ Questsion 2 -> (i) Part #######################################

    print("---------------------------- Q2 - part (i) Output------------------------")

    # Path of the dataset file
    dataset_file = 'dataset/dataset.csv'
    # load datasets from files
    dataset = pd.read_csv(dataset_file, sep=',', header=None)

    # # k = 4 and 5-different random initilization centroids
    # number_of_cluster = 4
    # kmeans1 = KMeans(n_cluster=number_of_cluster)

    # for i in range(1, 6):
    #     # Get the random centroid points everytime
    #     initial_centroids = common.random(dataset, number_of_cluster)
    #     clone = copy.deepcopy(dataset)

    #     kmeans1.fit(dataset,initial_centroids)
    #     kmeans1.save_figures('plots/Q2', number_of_cluster, 'Q2a{}'.format(i))

# ############################ Questsion 2 -> (ii) Part #######################################

    print("---------------------------- Q2 - part (ii) Output------------------------")
    
    # #fixed initialization and K = {2,3,4,5}

    # #initialization centroid for cluster
    # initial_centroids = ce.random(dataset, 2)

    # for i in range(2, 6):

    #     kmeans1 = KMeans(n_cluster=i)
    #     clone = copy.deepcopy(dataset)
    #     initial_centroids = common.add(clone, i)

    #     kmeans1.fit(dataset,initial_centroids)
    #     kmeans1.save_figures('plots/Q2', 4, 'Q2b{}'.format(i))

# ############################ Questsion 2 -> (iii) Part #######################################
    print("---------------------------- Q2 - part (iii) Output------------------------")
    
    import numpy as np
    file = open("dataset/Dataset.csv")
    mat3 = np.loadtxt(file, delimiter=",")

    mat3=np.transpose(mat3)
    import utilities.SpectralClustering as  sc

    affinity1 = sc.compute_affinity(mat3.T)

    k1 = sc.spectral_clustering(affinity1, 4)
    sc.plot_clusters(mat3, k1, 4, "Spectral on mat3, k=4")


# ############################ Questsion 2 -> (iv) Part #######################################
    print("---------------------------- Q2 - part (iv) Output------------------------")
    # data_df =dataset
    # data_df['cluster'] = sc.spectral_clustering(df=data_df, n_neighbors=8, n_clusters=3)

    # fig, ax = plt.subplots()
    # sns.scatterplot(x='x', y='y', data=data_df, hue='cluster', ax=ax)
    # ax.set(title='Spectral Clustering')

    print("---------------------------- Processing completed ------------------------")
   