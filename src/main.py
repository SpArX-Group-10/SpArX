from sklearn.model_selection import train_test_split

from clustering import KMeansClusterer
from model_encoder import Framework, Model
from ffnn import FFNN
from user_input import load_preset_dataset, get_ffnn_model_general, net_train
from merging import LocalMerger


def model_setup(dataset_name, activations, hidden_layers):
    """Model setup function."""

    # load data
    xtrain, ytrain = load_preset_dataset(dataset_name)

    # construct model
    model = get_ffnn_model_general(xtrain, ytrain, activations, hidden_layers)

    # split data
    x_train, x_test, y_train, y_test = train_test_split(
        xtrain, ytrain, test_size=0.1, random_state=42
    )

    # train model
    net_train(model, x_train, y_train, x_test, y_test, epochs=10)

    return model, x_train, x_test, y_train, y_test


def main():
    """Main function."""

    # parameters
    path = "breast cancer"
    activaitons = ["relu", "relu"]
    hidden_layers = [10, 10]

    # setup model
    model, x_train, _, _, _ = model_setup(path, activaitons, hidden_layers)

    # encode model
    encoded_model = Model.transform(model, Framework.KERAS)

    layer_shape = [x_train.shape[1]] + [size for [_, size] in encoded_model.layer_shapes]


    # custom model
    base_ffnn = FFNN(
        layer_shape, encoded_model.weights, encoded_model.biases, encoded_model.activation_functions
    )

    # forwardpass some data to do clustering
    base_ffnn.forward_pass(x_train.to_numpy())

    # create clusters
    cluster_labels = KMeansClusterer.cluster(base_ffnn, 0.5)

    # merge clusters
    merged_model = LocalMerger.merge(base_ffnn, cluster_labels)

    # print model summary
    base_ffnn.model.summary()
    merged_model.model.summary()


if __name__ == "__main__":
    main()
