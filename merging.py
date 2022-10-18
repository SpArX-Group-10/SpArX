from abc import abstractmethod

import numpy as np
from ffnn import FFNN


class Merger:
    """Abstract class for merging algorithms."""

    @staticmethod
    @abstractmethod
    def merge(mlp: FFNN, cluster_labels: np.ndarray) -> FFNN:
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
    def merge(mlp: FFNN, cluster_labels: np.ndarray) -> FFNN:
        merged_weights = []
        merged_biases = []

        partial_weights = [mlp.model.layers[0].get_weights()[0]]
        print(np.asarray(partial_weights).shape)

        new_shape = []

        for index in range(len(mlp.shape[1:-1])):

            nclusters = max(cluster_labels[index]) + 1
            new_shape.append(nclusters)

            # compute the new weights of the clustered layer
            merged_weights.append(GlobalMerger._merge_weights(index, nclusters, cluster_labels, partial_weights[index]))
            # compute the new biases of the clustered layer
            merged_biases.append(GlobalMerger._merge_biases(index, mlp, nclusters, cluster_labels))
            # update the intermediate weights (from merged layer to unmerged layer)
            partial_weights.append(GlobalMerger._update_partial_weights(index, mlp, nclusters, cluster_labels))

        merged_weights.append(partial_weights[-1])
        merged_biases.append(mlp.model.layers[len(mlp.shape[1:-1])].get_weights()[1])


        new_shape = [mlp.shape[0]] + new_shape + [mlp.shape[-1]]

        return FFNN(new_shape, merged_weights, merged_biases, mlp.activation_functions)


    @staticmethod
    def _merge_weights(
        index: int, num_clusters: int, cluster_labels: np.ndarray, partial_weights: np.ndarray
    ) -> list[np.ndarray]:
        """Merges the incoming weights of the given layer, according to the given cluster labels."""
        # print(f" partial weights before : {partial_weights.shape}")

        new_layer_weights = []
        for label in range(num_clusters):
            # TODO: test if we can use label instead of cluster_labels[index] == label
            new_layer_weights.append(np.mean(partial_weights.T[cluster_labels[index] == label], axis=0))

        # print(f" new weights after : {np.asarray(new_layer_weights).T.shape}")
        return np.asarray(new_layer_weights).T


# 4 -> 2
# [[a, b, c, d],  (2, 4)
#  [e, f, g, h]]

    @staticmethod
    def _merge_biases(
        index: int, mlp : FFNN, num_clusters:int, cluster_labels: np.ndarray
    ) -> list[np.ndarray]:
        """Merges the incoming bias weights of the given layer, according to the given cluster labels."""

        initial_biases = mlp.model.layers[index].get_weights()[1]
        new_layer_biases = []
        for label in range(num_clusters):
            new_layer_biases.append(np.mean(initial_biases[cluster_labels[index] == label]))

        return np.asarray(new_layer_biases)


    @staticmethod
    def _update_partial_weights(
        index: int, mlp: FFNN, num_clusters:int, cluster_labels: np.ndarray
    ) -> np.ndarray:
        """Updates the intermediate weights (from merged layer to unmerged layer)."""
        print(f"old weight : {mlp.model.layers[index + 1].get_weights()[0].shape}")

        new_partial_weights = []

# outgoing_weights[index + 1].append(
    # (np.sum(
    #       model.layers[index + 1].get_weights()[0][clustering_labels[index] == label], axis=0))
    # .reshape((1, -1))
    # )

        for label in range(num_clusters):
            new_partial_weights.append(
                np.sum(mlp.model.layers[index + 1].get_weights()[0][cluster_labels[index] == label], axis=0))

        print(f"new partial weight : {np.asarray(new_partial_weights).shape}")

        return np.asarray(new_partial_weights)


class LocalMerger(Merger):
    """Merges the given model using the local merge technique."""

    @staticmethod
    def merge(mlp: FFNN, cluster_labels: np.ndarray) -> FFNN:
        raise NotImplementedError
