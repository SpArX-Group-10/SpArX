from abc import abstractmethod
from typing import Optional
from math import ceil
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from .ffnn import FFNN


class Clusterer:
    """Abstract class for clustering algorithms."""

    @staticmethod
    @abstractmethod
    def cluster_layer(num_clusters: int, data: np.ndarray, seed: Optional[int]=None) -> np.ndarray:
        """Clusters the given layer

        :param num_clusters: the number of clusters to create
        :type num_clusters: int
        :param data: the data to cluster from each layer
        :type data: np.ndarray
        :param seed: the seed to use to cluster the current layer, defaults to None
        :type seed: Optional[int], optional
        :raises NotImplementedError:
        :return: the label for that layer
        :rtype: np.ndarray
        """
        raise NotImplementedError


    @staticmethod
    def _relabel_cluster(clusters_labels: np.ndarray) -> np.ndarray:
        """Relabel the cluster labels to avoid gaps within the labels
        """
        _, new_labels = np.unique(clusters_labels, return_inverse=True)
        return new_labels


    @classmethod
    def cluster(cls, mlp: FFNN, shrink_to_percentage: float, seed: Optional[int]=None) -> list[np.ndarray]:
        """Clusters the given model

        :param mlp: the mlp model to be clustered
        :type mlp: FFNN
        :param shrink_to_percentage: the percentatge of the original size to shrink to
        :type shrink_to_percentage: float
        :param seed: the seed to use for the clustering algorithm, defaults to None
        :type seed: Optional[int], optional
        :return: a list of labels for each layer of the network
        :rtype: list[np.ndarray]
        """        

        clustering_labels = []
        ffnn_shape = mlp.get_shape()
        for index in range(1, len(mlp.model.layers)):
            num_clusters = int(ceil(shrink_to_percentage * ffnn_shape[index]))
            activation = mlp.forward_pass_data[index - 1]
            clustering_input = activation.T

            unsanitised_cluster = cls.cluster_layer(num_clusters, clustering_input, seed)
            relabled_cluster = Clusterer._relabel_cluster(unsanitised_cluster)

            clustering_labels.append(relabled_cluster)


        return clustering_labels


class KMeansClusterer(Clusterer):
    """KMeans clustering algorithm."""

    @staticmethod
    def cluster_layer(num_clusters: int, data: np.ndarray, seed: Optional[int]=None) -> np.ndarray:
        return KMeans(n_clusters=num_clusters, random_state=seed).fit(data).labels_


class AgglomerativeClusterer(Clusterer):
    """Agglomerative clustering algorithm."""

    @staticmethod
    def cluster_layer(num_clusters: int, data: np.ndarray, seed: Optional[int]=None) -> np.ndarray:
        return AgglomerativeClustering(n_clusters=num_clusters).fit(data).labels_
