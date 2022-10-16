from abc import abstractmethod

import numpy as np
from ffnn import FFNN


class Merger:
    """Abstract class for merging algorithms."""

    @abstractmethod
    @staticmethod
    def merge(cluster_labels: np.ndarray, mlp: FFNN) -> FFNN:
        """Merges the given model.

        :param
            cluster_labels: np.ndarray
                the labels of the clusters.
            mlp: FFNN
                the model to merge.
        """
        raise NotImplementedError


class GlobalMerger(Merger):
    """Merges the given model using the global merge technique."""


    @classmethod
    @staticmethod
    def merge(cls, cluster_labels: np.ndarray, mlp: FFNN) -> FFNN:
        merged_weights = []
        merged_biases = []

        partial_weights = [[mlp.model.layers[0].get_weights()[0]]]  

        for index in range(len(mlp.shape[1:-1])):

            nclusters = max(cluster_labels[index])

            # compute the new weights of the clustered layer
            merged_weights.append(cls._merge_weights(index, nclusters, cluster_labels, partial_weights[index]))
            # compute the new biases of the clustered layer
            merged_biases.append(cls._merge_biases(index, mlp, nclusters, cluster_labels))
            # update the intermediate weights (from merged layer to unmerged layer)
            partial_weights.append(cls._update_partial_weights(index, mlp, nclusters, cluster_labels))


        merged_weights.append(partial_weights[-1])
        merged_biases.append([mlp.layers[len(mlp.shape[1:-1])].get_weights()[1]])

        return merged_weights, merged_biases


    @classmethod
    def _merge_weights(
        cls, index: int, num_clusters: int, cluster_labels: np.ndarray, partial_weights: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Merges the incoming weights of the given layer, according to the given cluster labels."""

        new_layer_weights = []
        for label in range(num_clusters):
            #TODO: test if we can use label instead of cluster_labels[index] == label
            new_layer_weights.append(np.mean(np.vstack(partial_weights).T[cluster_labels[index] == label], axis=0))

        return new_layer_weights


    @classmethod
    def _merge_biases(
        cls, index: int, mlp : FFNN, num_clusters:int, cluster_labels: np.ndarray
    ) -> list[np.ndarray]:
        """Merges the incoming bias weights of the given layer, according to the given cluster labels."""

        initial_biases = mlp.layers[index].get_weights[1]
        new_layer_biases = []
        for label in range(num_clusters):
            new_layer_biases.append(np.mean(initial_biases[cluster_labels == label]))

        return new_layer_biases

    @classmethod
    def _update_partial_weights(
        cls, index: int, mlp : FFNN, num_clusters:int, cluster_labels: np.ndarray
    ) -> list[np.ndarray]:
        """Updates the intermediate weights (from merged layer to unmerged layer)."""

        new_partial_weights = []
        for label in range(num_clusters):
            new_partial_weights.append(np.sum(
                mlp.model.layers[index + 1].get_weights()[0][cluster_labels[index] == label], axis=0)).reshape((1, -1))

        return new_partial_weights


class LocalMerger(Merger):
    """Merges the given model using the local merge technique."""

    @staticmethod
    def merge(cluster_labels: np.ndarray, mlp: FFNN) -> FFNN:
        raise NotImplementedError
