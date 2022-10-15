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

    @staticmethod
    def merge(cluster_labels: np.ndarray, mlp: FFNN) -> FFNN:
        # merged_weights = []
        # merged_biases = []
        # for index in range(len(mlp.shape[1:-1])):
        #     num_clusters = max(cluster_labels[index])

        #     merged_weights.append(self.merge_weights(index, mlp, num_clusters))
        #     merged_biases.append(self.merge_biases(index, mlp, num_clusters))

        raise NotImplementedError

    def _merge_weights(
        self, layer_index: int, mlp: FFNN, num_clusters: int
    ) -> np.ndarray:
        # new_layer_weights = []
        # for label in range(num_clusters):
        #      new_layer_weights.append(np.mean(np.vstack(mlp.model.layers[layer_index]
        #           .get_weights()[0]).T[clustering_labels[index] == label], axis=0))
        pass


class LocalMerger(Merger):
    """Merges the given model using the local merge technique."""

    @staticmethod
    def merge(cluster_labels: np.ndarray, mlp: FFNN) -> FFNN:
        raise NotImplementedError
