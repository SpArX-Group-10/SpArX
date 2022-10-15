from abc import abstractmethod
import numpy as np
import sklearn
from ffnn import FFNN


class Clusterer:
    """Abstract class for clustering algorithms."""

    @staticmethod
    @abstractmethod
    def cluster_layer(num_clusters: int, data: np.ndarray) -> np.ndarray:
        """Clusters the given model.

        :param
            num_clusters: int
                the number of clusters to create.
            data: np.ndarray
                the data to cluster from each layer.
        """
        raise NotImplementedError


    @classmethod
    def cluster(cls, num_clusters: int, mlp: FFNN) -> np.ndarray:
        """Clusters the given model.

        :param
            num_clusters: int
                the number of clusters to create.
            data: np.ndarray
                the data to cluster.
        """

        clustering_labels = []
        for index in range(1, len(mlp.shape) - 1):
            activation = mlp.forward_pass_data[index - 1]
            clustering_input = activation.T
            clustering_labels.append(cls.cluster_layer(num_clusters, clustering_input))
        return clustering_labels


class KMeansClusterer(Clusterer):
    """KMeans clustering algorithm."""

    @staticmethod
    def cluster_layer(num_clusters: int, data: np.ndarray) -> np.ndarray:
        return sklearn.cluster.KMeans(n_clusters=num_clusters).fit(data).labels_


class AgglomerativeClusterer(Clusterer):
    """Agglomerative clustering algorithm."""

    @staticmethod
    def cluster_layer(num_clusters: int, data: np.ndarray) -> np.ndarray:
        return sklearn.cluster.AgglomerativeClustering(n_clusters=num_clusters).fit(data).labels_
