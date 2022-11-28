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
        """Clusters the given model.

        :param
            num_clusters: int
                the number of clusters to create.
            data: np.ndarray
                the data to cluster from each layer.
            seed: Optional[int]
                the seed to use to cluster the current layer

        :returns
            the label for that layer
        """
        raise NotImplementedError


    @staticmethod
    def _relabel_cluster(clusters_labels: np.ndarray) -> np.ndarray:
        """Relable the cluster labels to avoid gaps within the labels

        :params
            clusters_labels: np.ndarray
                the cluster to be relabeld

        :returns
            reclustered labels
        """
        _, new_labels = np.unique(clusters_labels, return_inverse=True)
        return new_labels


    @classmethod
    def cluster(cls, mlp: FFNN, shrink_to_percentage: float, seed: Optional[int]=None) -> list[np.ndarray]:
        """Clusters the given model.

        :param
            mlp: FFNN
                the mlp model to be clustered
            shrink_to_percentage: float
                the percentatge of the original size to shrink to.
            data: np.ndarray
                the data to cluster.
            seed: Optional
                the seed to use for the clustering algorithm.

        :returns
            a list of labels for each layer of the network
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
