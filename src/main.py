import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

from clustering import KMeansClusterer
from model_encoder import Framework, Model
from ffnn import FFNN
from user_input import import_dataset, get_ffnn_model_general, net_train
from merging import LocalMerger
from visualiser import SimpleVisualizer


# parameters
PATH = "data/iris.data"

ACTIVATIONS = ["relu", "relu"]
HIDDEN_LAYERS = [10, 10]
OUTPUT_ACTIVATION = "softmax"

SRHINK_TO_FACTOR = 0.5
MERGING_DATA_POINT = [[4.9, 3.1, 1.5, 0.1]]

CLUSTERER = KMeansClusterer
MERGER = LocalMerger
VISUALISER = SimpleVisualizer


def model_setup(dataset_name, activations, hidden_layers):
    """Model setup function."""

    # load data
    xtrain, ytrain = import_dataset(dataset_name, has_index=False)

    # if y is a string, convert to 1 hot encoding
    if ytrain.dtypes[0] == "object":
        ytrain = pd.get_dummies(ytrain)

    # construct model
    model = get_ffnn_model_general(xtrain, ytrain, activations, hidden_layers, OUTPUT_ACTIVATION)

    # split data
    x_train, x_test, y_train, y_test = train_test_split(
        xtrain, ytrain, test_size=0.1, random_state=42
    )

    # train model
    net_train(model, x_train, y_train, x_test, y_test, epochs=10)

    input_names = xtrain.columns.to_numpy().tolist()
    output_names = ytrain.columns.to_numpy().tolist()

    return model, x_train, input_names, output_names


def main():
    """Main function."""
    full_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), PATH)



    # phase 1

    # setup model
    model, x_train, input_names, _ = model_setup(full_path, ACTIVATIONS, HIDDEN_LAYERS)

    # encode model
    encoded_model = Model.transform(model, Framework.KERAS)

    layer_shape = [x_train.shape[1]] + [size for [_, size] in encoded_model.layer_shapes]

    # custom model
    base_ffnn = FFNN(layer_shape, encoded_model.weights, encoded_model.biases, encoded_model.activation_functions)



    # Phase 2

    # forwardpass some data to do clustering
    base_ffnn.forward_pass(x_train.to_numpy())

    # create clusters
    cluster_labels = CLUSTERER.cluster(base_ffnn, SRHINK_TO_FACTOR)

    if MERGING_DATA_POINT:
        base_ffnn.forward_pass(np.array(MERGING_DATA_POINT))

    # merge clusters
    merged_model = MERGER.merge(base_ffnn, cluster_labels)


    # Phase 3
    return VISUALISER.visualise(merged_model, input_names)


if __name__ == "__main__":
    main()
