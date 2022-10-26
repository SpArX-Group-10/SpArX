import pytest
import keras  # pylint: disable=import-error
from keras.models import Sequential  # pylint: disable=import-error
from keras.layers import Dense, Input  # pylint: disable=import-error
from model_encoder import Model, Framework

def test_get_keras_model_info():
    """Test model info encoding."""
    ff_layers = [
        Input(shape=(5,)),
        Dense(2, activation='relu'),
    ]
    model = Sequential(ff_layers)
    res = Model.get_keras_model_info(model)
    assert res.num_layers == 1
    assert res.layer_shapes == [(None, 2)]
    assert len(res.weights) == 1
    assert res.weights[0].shape == (5, 2)
    assert len(res.biases) == 1
    assert res.biases[0].shape == (2,)
    assert res.activation_functions == ['relu']


def test_activation_to_str():
    """Test translate function from activation function to string."""
    assert Model.activation_to_str(keras.activations.softmax) == "softmax"
    assert Model.activation_to_str(keras.activations.relu) == "relu"
    assert Model.activation_to_str(keras.activations.tanh) == "tanh"

    with pytest.raises(NotImplementedError):
        Model.activation_to_str("Not an activation function")


def test_transform():
    """Test transform"""
    ff_layers = [
        Input(shape=(5,)),
        Dense(2, activation='relu'),
    ]
    model = Sequential(ff_layers)

    with pytest.raises(NotImplementedError):
        Model.transform(model, Framework.PYTORCH)
