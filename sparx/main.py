import numpy as np

from .model_encoder import Model
from .ffnn import FFNN
from .merging import LocalMerger


def main(xdata, model, framework, clusterer, merger, visualiser, datapoint, shrink_factor):
    """Main function."""

    # phase 1

    # encode model
    encoded_model = Model.transform(model, framework)

    layer_shape = tuple([xdata.shape[1]] + [size for [_, size] in encoded_model.layer_shapes])

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
