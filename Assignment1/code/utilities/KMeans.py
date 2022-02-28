import numpy as np
import matplotlib.pyplot as plt
import copy
import os

class KMeans:
    def __init__(self, n_cluster=3, random_state=721):
        self.n_cluster = n_cluster
        self.random_state = random_state

    def fit(self, dataset,initial_centroids):
        self.X = dataset.iloc[:, [0, 1]]  # not use feature labels
        self.m = self.X.shape[0]  # number of training examples
        self.n = self.X.shape[1]  # number of features.    
        initial_centroids = initial_centroids
        
        self.plot_initial_centroids(initial_centroids)
        self.clustering(initial_centroids)

    def plot_initial_centroids(self, initial_centroids):

        plt.scatter(self.X.iloc[:,0], self.X.iloc[:,1], c='#000000', s=7, label='Data Points')
        plt.scatter(initial_centroids[:, 0], initial_centroids[:, 1], marker='*', s=120, c='r', label='Initial Centroids')
        plt.title('Initial Random Cluster Centers')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.draw()

    def clustering(self, centroids):

        old_centroids = np.zeros(centroids.shape)
        stopping_criteria = 0.0001
        self.iterating_count = 0
        self.objective_func_values = []

        while self.euclidean_distance(old_centroids, centroids) > stopping_criteria:
            clusters = np.zeros(len(self.X))
            # Assigning each value to its closest cluster
            for i in range(self.m):
                distances = []
                for j in range(len(centroids)):
                    distances.append(self.euclidean_distance(self.X.iloc[i, :], centroids[j]))
                cluster = np.argmin(distances)
                clusters[i] = cluster

            # Storing the old centroid values to compare centroid moves
            old_centroids = copy.deepcopy(centroids)

            # Finding the new centroids
            for i in range(self.n_cluster):
                points = [self.X.iloc[j, :] for j in range(len(self.X)) if clusters[j] == i]
                centroids[i] = np.mean(points, axis=0)

            # calculate objective function value for current cluster centroids
            self.objective_func_values.append([self.iterating_count, self.objective_func_calculate(clusters, centroids)])
            
            self.iterating_count += 1

        self.plot_objective_function_values()
        self.plot_centroids(centroids, clusters)
        #plt.savefig('plots\Q2\A.png', dpi=300)


    def plot_centroids(self, centroids, clusters):
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
          "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
        fig, ax = plt.subplots()
        for i in range(self.n_cluster):
            points = np.array([self.X.iloc[j, :] for j in range(len(self.X)) if clusters[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i], label='Cluster {}'.format(i + 1))
        ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=120, c='#000000', label='Centroids')

        plt.title('k-Means Clustering\n( Iteration count = {} Objective Function value = {:.2f} )'
                  .format((self.iterating_count), np.array(self.objective_func_values)[self.iterating_count-1, 1]))
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.draw()

    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((np.array(a) - np.array(b))**2))

    def objective_func_calculate(self, clusters, centroids):
        """Calculate objective function value for current centroids"""

        # Calculate objective function value
        distances_from_centroids = []
        for i in range(self.n_cluster):
            points = np.array([self.X.iloc[j, :] for j in range(len(self.X)) if clusters[j] == i])
            for k in range(len(points)):
                distances_from_centroids.append(self.euclidean_distance(points[k, :], centroids[i]))
        return sum(distances_from_centroids)

    def plot_objective_function_values(self):
        """This function plot graph of objective function value for each iteration """

        plt.figure()
        plt.plot((np.array(self.objective_func_values)[:, 0] + 1),  np.array(self.objective_func_values)[:, 1], 'bo')
        plt.plot((np.array(self.objective_func_values)[:, 0] + 1), np.array(self.objective_func_values)[:, 1], 'k')
        plt.title('Objective Function')
        plt.xlabel('Iteration Number')
        plt.ylabel('Objective Function Value')
        plt.draw()

    def save_figures(self, path,k,question_number):
        """Save all figures plotted with matplotlib to path directory"""

        # create folder for png files
        if not os.path.isdir(path):
            os.makedirs(path)

        # plt.get_fignums returns a list of existing figure numbers.
        # then we save all existing figures
        for i in plt.get_fignums():
            plt.figure(i)
            plt.savefig(os.path.join(path, "{}_K{}_{}.png".format(question_number,k,i)), format='png')
        # close all figure to clear figure numbers        
        plt.close("all")
        plt.clf()

   