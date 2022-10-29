from this import d
import pytest
from keras.models import Sequential # pylint: disable=import-error
from keras.layers import Activation, Dense, Input # pylint: disable=import-error
from src.user_input import import_dataset, import_model, Framework, verify_keras_model_is_fnn # pylint: disable=import-error

# Testing approach 1: importing a model
def test_import_model_keras_and_fnn():
    """ Import a Keras FFNN model. """
    ff_layers = [
        Dense(10, activation='relu'),
        Dense(2, activation='softmax')
    ]
    model = Sequential(ff_layers)
    assert import_model(Framework.KERAS, model) == model


def test_import_unsupported_model():
    """ Fails when importing unsupported models. """
    model = "Not Keras Model!"

    with pytest.raises(ValueError) as exc_info:
        import_model(Framework.KERAS, model)
    assert str(exc_info.value) == "Model is not a Keras model."


def test_import_model_keras_and_not_fnn():
    """ Fails when importing a Keras model that is not FFNN. """
    ff_layers = [
        Input(shape=(10,)),
        Dense(10, activation='relu'),
        Dense(2, activation='softmax'),
        Activation('relu')
    ]
    model = Sequential(ff_layers)

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
    ff_layers = [
        Dense(10, activation='relu'),
        Dense(2, activation='softmax'),
    ]
    model = Sequential(ff_layers)

    assert verify_keras_model_is_fnn(model)


def test_verify_model_is_not_fnn():
    """ Returns false when Keras model that is not FFNN. """
    ff_layers = [
        Input(shape=(10,)),
        Dense(10, activation='relu'),
        Dense(2, activation='softmax'),
        Activation('relu')
    ]
    model = Sequential(ff_layers)

    assert not verify_keras_model_is_fnn(model)


# Tests for approach 2: training a model given parameters
def test_import_dataset():
    """ Imports raw dataset and process the data. """
    filepath = "test/data/test_user_input_data.csv"
    data_entries, labels = import_dataset(filepath)
    assert data_entries.shape == (3, 2)
    assert labels.shape == (3,)


def test_import_dataset_features():
    """ Imports raw dataset with selected features. """
    filepath = "test/data/test_user_input_data.csv"
    data_entries, labels = import_dataset(filepath, ["name"])
    assert data_entries.shape == (3, 1)
    assert labels.shape == (3,)


# def test_train_model():
#     self.assertTrue(False and "not implemented")

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
