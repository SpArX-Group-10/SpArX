from abc import abstractmethod
import numpy as np
from .ffnn import FFNN


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

        new_shape = []

        num_hidden_layers = len(mlp.model.layers) - 1

        for index in range(num_hidden_layers):

            nclusters = max(cluster_labels[index]) + 1
            new_shape.append(nclusters)

            # compute the new weights of the clustered layer
            merged_weights.append(GlobalMerger._merge_weights(index, nclusters, cluster_labels, partial_weights[index]))
            # compute the new biases of the clustered layer
            merged_biases.append(GlobalMerger._merge_biases(index, mlp, nclusters, cluster_labels))
            # update the intermediate weights (from merged layer to unmerged layer)
            partial_weights.append(GlobalMerger._update_partial_weights(index, mlp, nclusters, cluster_labels))

        merged_weights.append(partial_weights[-1])
        merged_biases.append(mlp.model.layers[num_hidden_layers].get_weights()[1])


        new_shape = [mlp.model.layers[0].input_shape[1]] + new_shape + [mlp.model.layers[-1].output_shape[1]]
        return FFNN(new_shape, merged_weights, merged_biases, mlp.activation_functions)


    @staticmethod
    def _merge_weights(
        index: int, num_clusters: int, cluster_labels: np.ndarray, partial_weights: np.ndarray
    ) -> list[np.ndarray]:
        """Merges the incoming weights of the given layer, according to the given cluster labels."""

        new_layer_weights = []
        for label in range(num_clusters):
            # TODO: test if we can use label instead of cluster_labels[index] == label
            new_layer_weights.append(np.mean(partial_weights.T[cluster_labels[index] == label], axis=0))

        return np.asarray(new_layer_weights).T


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

        new_partial_weights = []

        for label in range(num_clusters):
            new_partial_weights.append(
                np.sum(mlp.model.layers[index + 1].get_weights()[0][cluster_labels[index] == label], axis=0))

        return np.asarray(new_partial_weights)


class LocalMerger(Merger):
    """Merges the given model using the global merge technique."""

    @staticmethod
    def merge(mlp: FFNN, cluster_labels: np.ndarray) -> FFNN:
        merged_weights = []
        merged_biases = []

        partial_weights = [mlp.model.layers[0].get_weights()[0]]

        old_shape = mlp.get_shape()

        # create empty shrunken model
        smlp = FFNN((old_shape[0], ), [], [], [])
        smlp.data = mlp.data
        new_shape = []

        for index in range(len(old_shape[1:-1])):

            nclusters = max(cluster_labels[index]) + 1
            new_shape.append(nclusters)

            # compute the new weights of the clustered layer
            merged_weights.append(LocalMerger._merge_weights(index, nclusters, cluster_labels, partial_weights[index]))
            # compute the new biases of the clustered layer
            merged_biases.append(LocalMerger._merge_biases(index, mlp, nclusters, cluster_labels))
            # update the intermediate weights (from merged layer to unmerged layer)

            # TODO: set up shrunken mlp layer
            smlp.add_layer(nclusters, merged_weights[index], merged_biases[index], mlp.activation_functions[index])

            partial_weights.append(LocalMerger._update_partial_weights(index, mlp, smlp, nclusters, cluster_labels))

        merged_weights.append(partial_weights[-1])
        merged_biases.append(mlp.model.layers[len(old_shape[1:-1])].get_weights()[1])


        new_shape = [old_shape[0]] + new_shape + [old_shape[-1]]

        return FFNN(new_shape, merged_weights, merged_biases, mlp.activation_functions)


    @staticmethod
    def _merge_weights(
        index: int, num_clusters: int, cluster_labels: np.ndarray, partial_weights: np.ndarray
    ) -> list[np.ndarray]:
        """Merges the incoming weights of the given layer, according to the given cluster labels."""

        new_layer_weights = []
        for label in range(num_clusters):
            # TODO: test if we can use label instead of cluster_labels[index] == label
            new_layer_weights.append(np.mean(partial_weights.T[cluster_labels[index] == label], axis=0))

        return np.asarray(new_layer_weights).T


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
        index: int, mlp: FFNN, smlp: FFNN, num_clusters:int, cluster_labels: np.ndarray
    ) -> np.ndarray:
        """Updates the intermediate weights (from merged layer to unmerged layer)."""

        new_partial_weights = []
        for label in range(num_clusters):
            # this is cached
            h_star = smlp.forward_pass_data[index][:, label]

            all_hidden_activations = mlp.forward_pass_data[index][:, cluster_labels[index] == label]
            h_star = np.array([1 if h_s == 0 else h_s for h_s in list(h_star)])
            normalized_activations = np.sum(np.divide(all_hidden_activations.T, h_star).T, axis = 0)

            new_partial_weights.append(
                np.dot(
                    normalized_activations,
                    mlp.model.layers[index + 1].get_weights()[0][cluster_labels[index] == label]
                )
            )


        return np.asarray(new_partial_weights)
