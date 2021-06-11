import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances_argmin, silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class Kmeans:

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.dimensions = len(data[0])
        self.centers = None
        self.labels = None

    def find_clusters(self, num_clusters, rseed=2):

        rng = np.random.RandomState(rseed)
        i = rng.permutation(self.data.shape[0])[:num_clusters]
        self.centers = self.data[i]

        while True:
            self.labels = pairwise_distances_argmin(self.data, self.centers)
            new_centers = np.array([self.data[self.labels == i].mean(0)
                                    for i in range(num_clusters)])
            if np.all(self.centers == new_centers):
                break
            self.centers = new_centers
        return self.labels

    def plot_kmeans(self):
        if self.dimensions == 2:
            self.plot_2d_kmeans()
        else:
            self.plot_3d_kmeans()
        return

    def plot_2d_kmeans(self):
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels,
                    linewidth=0, antialiased=False)

        plt.axis('equal')
        plt.show()
        return

    def plot_3d_kmeans(self, view_init=None):
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.data[:, 0], self.data[:, 1], self.data[:, 2], c=self.labels,
                     linewidth=0, antialiased=False)
        if view_init:
            ax.view_init(view_init[0], view_init[1])
        plt.show()
        return
