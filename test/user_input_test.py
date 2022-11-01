import pytest
from keras.models import Sequential # pylint: disable=import-error
from keras.layers import Activation, Dense, Input # pylint: disable=import-error
from sklearn.model_selection import train_test_split # pylint: disable=import-error
from user_input import import_dataset, import_model, Framework, verify_keras_model_is_fnn # pylint: disable=import-error
from user_input import train_model, get_ffnn_model_general, net_train # pylint: disable=import-error


# Testing approach 1: importing a model
def keras_fnn_model():
    """ Returns a keras ffnn model. """
    ff_layers = [
        Dense(10, activation='relu'),
        Dense(2, activation='softmax')
    ]
    return Sequential(ff_layers)

def keras_not_fnn_model():
    """ Returns a keras model that is not a ffnn. """
    ff_layers = [
        Input(shape=(10,)),
        Dense(10, activation='relu'),
        Dense(2, activation='softmax'),
        Activation('relu')
    ]
    return Sequential(ff_layers)


def test_import_model_keras_and_fnn():
    """ Import a Keras FFNN model. """
    model = keras_fnn_model()

    assert import_model(Framework.KERAS, model) == model


def test_import_unsupported_model():
    """ Fails when importing unsupported models. """
    model = "Not Keras Model!"

    with pytest.raises(ValueError) as exc_info:
        import_model(Framework.KERAS, model)
    assert str(exc_info.value) == "Model is not a Keras model."


def test_import_model_keras_and_not_fnn():
    """ Fails when importing a Keras model that is not FFNN. """
    model = keras_not_fnn_model()

    with pytest.raises(ValueError) as exc_info:
        import_model(Framework.KERAS, model)
    assert str(exc_info.value) == "Model is not a feed-forward neural network."


def test_import_model_unsupported_framework():
    """ Fails when importing unsupported framework. """
    with pytest.raises(ValueError) as exc_info:
        import_model("Not a framework", None)
    assert str(exc_info.value) == "Unsupported framework!"


def test_verify_model_is_fnn():
    """ Returns true when Keras model that is FFNN. """
    model = keras_fnn_model()

    assert verify_keras_model_is_fnn(model)


def test_verify_model_is_not_fnn():
    """ Returns false when Keras model that is not FFNN. """
    model = keras_not_fnn_model()

    assert not verify_keras_model_is_fnn(model)


# Tests for approach 2: training a model given parameters
USER_INPUT_DATA_FILEPATH = "test/data/test_user_input_data.csv"

def check_layers(model, activation_functions, hidden_layers_size):
    """ Check that the model has the correct number of layers: input layer,
    hidden layers (as specified in given parameters), and output layer. """
    assert len(model.layers) == len(hidden_layers_size) + 2

    # Check that each hidden layer has the activation function and number of neurons as specified in the parameters
    for i, (hidden_layer_size, activation_func) in enumerate(zip(hidden_layers_size, activation_functions)):
        assert model.layers[i + 1].get_config()["units"] == hidden_layer_size
        assert model.layers[i + 1].get_config()["activation"] == activation_func


def test_import_dataset():
    """ Imports raw dataset and process the data. """
    data_entries, labels = import_dataset(USER_INPUT_DATA_FILEPATH)
    # Check that the data and labels are imported correctly
    assert data_entries.shape == (3, 2)
    assert labels.shape == (3, 1)


def test_import_dataset_features_one_feature():
    """ Imports raw dataset with selected features. """
    data_entries, labels = import_dataset(USER_INPUT_DATA_FILEPATH, ["age"])
    # Check the feature is imported correctly
    assert data_entries.shape == (3, 1)
    assert labels.shape == (3, 1)


def test_import_dataset_features_multiple_features():
    """ Imports raw dataset with selected features. """
    data_entries, labels = import_dataset(USER_INPUT_DATA_FILEPATH, ["age", "height"])
    # Check the feature is imported correctly
    assert data_entries.shape == (3, 2)
    assert labels.shape == (3, 1)


# def test_import_dataset_nonexistent_file():
#     # TODO
#     """ Fails when importing a dataset from a nonexistent file. """
#     with pytest.raises(FileNotFoundError) as exc_info:
#         import_dataset("nonexistent_file.csv")
#     assert str(exc_info.value) == "File does not exist."
#     self.assertTrue(False and "not implemented")


# def test_import_dataset_incorrect_features():
#     # TODO
#     self.assertTrue(False and "not implemented")


def test_get_general_ffnn_model():
    """ Test that the general FFNN model is created correctly. """
    # Train model with given parameters
    activation_functions = ["relu", "softmax"]
    hidden_layers_size = [2, 3]
    x_data, y_data = import_dataset(USER_INPUT_DATA_FILEPATH)
    model = get_ffnn_model_general(
        x_data, y_data, activation_functions, hidden_layers_size)

    # Check that each layer has the activation function and number of neurons as specified in the parameters
    check_layers(model, activation_functions, hidden_layers_size)


def test_get_general_ffnn_model_with_no_hidden_layers():
    """ Test that the general FFNN model is created correctly with no hidden layers. """
    # No activation functions and hidden layers size are specified
    activation_functions = []
    hidden_layers_size = []
    x_data, y_data = import_dataset(USER_INPUT_DATA_FILEPATH)
    model = get_ffnn_model_general(
        x_data, y_data, activation_functions, hidden_layers_size)

    # Model created when no hidden layers are specified
    assert len(model.layers) == 1
    assert model.layers[0].get_config()["activation"] == "softmax"


# def test_recall_m():
#     self.assertTrue(False and "not implemented")

# def test_precision_m():
#     self.assertTrue(False and "not implemented")


def test_net_train():
    """ Weights of the model are updated after training. """
    # Train model with given parameters
    activation_functions = ["relu", "softmax"]
    hidden_layers_size = [2, 3]
    x_data, y_data = import_dataset(USER_INPUT_DATA_FILEPATH)

    # Initial weights of the model before training
    model = get_ffnn_model_general(
        x_data, y_data, activation_functions, hidden_layers_size)
    weights_before_train = model.get_weights()

    # Weights of the model after training
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=.2, random_state=2, shuffle=True)
    net_train(model, x_train, y_train, x_test, y_test)
    weights_after_train = model.get_weights()

    # Check that the weights have changed
    assert len(weights_before_train) == len(weights_after_train)
    for before_layer_weights, after_layer_weights in zip(weights_before_train, weights_after_train):
        assert len(before_layer_weights) == len(after_layer_weights)
        assert not (before_layer_weights == after_layer_weights).all()


def test_train_model():
    """ Test layers of the models have attributes that are as given. """
    # Train model with given parameters
    activation_functions = ["relu", "softmax"]
    hidden_layers_size = [2, 3]
    model = train_model(USER_INPUT_DATA_FILEPATH, activation_functions, hidden_layers_size)

    # Check layers of model
    check_layers(model, activation_functions, hidden_layers_size)
