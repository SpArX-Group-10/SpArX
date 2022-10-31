import pytest
from keras.models import Sequential # pylint: disable=import-error
from keras.layers import Activation, Dense, Input # pylint: disable=import-error
from user_input import import_dataset, import_model, Framework, verify_keras_model_is_fnn # pylint: disable=import-error

# Testing approach 1: importing a model
def keras_fnn_model():
    """ Returns a keras fnn model. """
    ff_layers = [
        Dense(10, activation='relu'),
        Dense(2, activation='softmax')
    ]
    return Sequential(ff_layers)

def keras_not_fnn_model():
    """ Returns a keras model that is not a fnn. """
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

def test_import_dataset():
    """ Imports raw dataset and process the data. """
    data_entries, labels = import_dataset(USER_INPUT_DATA_FILEPATH)
    assert data_entries.shape == (3, 3)
    assert labels.shape == (3, 1)


def test_import_dataset_features():
    """ Imports raw dataset with selected features. """
    data_entries, labels = import_dataset(USER_INPUT_DATA_FILEPATH, ["name"])
    assert data_entries.shape == (3, 1)
    assert labels.shape == (3, 1)


# def test_train_model():
#     """ Train model with given parameters. """
#     filepath = "test/data/test_user_input_data.csv"
#     activation_functions = ["relu", "softmax"]
#     hidden_layers_size = [2, 3]
#     model = train_model(filepath, activation_functions, hidden_layers_size)


# def test_recall_m():
#     self.assertTrue(False and "not implemented")

# def test_precision_m():
#     self.assertTrue(False and "not implemented")

# def test_get_ffnn_model():
#     self.assertTrue(False and "not implemented")

# def test_get_general_ffnn_model():
#     self.assertTrue(False and "not implemented")

# def test_net_train():
#     self.assertTrue(False and "not implemented")
