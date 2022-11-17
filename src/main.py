import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

from .clustering import KMeansClusterer
from .model_encoder import Framework, Model
from .ffnn import FFNN
from .user_input import import_dataset, get_ffnn_model_general, net_train
from .merging import LocalMerger
from .visualiser import SimpleVisualizer


def main(xdata, model, framework, clusterer, merger, visualiser, datapoint, shrink_factor):
    """Main function."""

    # phase 1

    # encode model
    encoded_model = Model.transform(model, framework)

    layer_shape = [xdata.shape[1]] + [size for [_, size] in encoded_model.layer_shapes]

    # custom model
    base_ffnn = FFNN(layer_shape, encoded_model.weights, encoded_model.biases, encoded_model.activation_functions)

    # Phase 2

    # forwardpass some data to do clustering
    base_ffnn.forward_pass(xdata.to_numpy())

    # create clusters
    cluster_labels = clusterer.cluster(base_ffnn, shrink_factor)

    if merger is LocalMerger:
        base_ffnn.forward_pass(np.array([datapoint]))

    # merge clusters
    merged_model = merger.merge(base_ffnn, cluster_labels)

    # Phase 3
    return visualiser.visualise(merged_model, xdata.columns.to_numpy())


if __name__ == "__main__":
    main()
