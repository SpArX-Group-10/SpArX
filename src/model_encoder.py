from enum import Enum, auto

import keras


class Framework(Enum):
    """Framework enum"""

    KERAS = auto()
    PYTORCH = auto()


class EncodedModel:
    """Encoded model class."""

    def __init__(self, num_layers, layer_shapes, weights, biases, activation_functions):
        self.num_layers = num_layers
        self.layer_shapes = layer_shapes
        self.weights = weights
        self.biases = biases
        self.activation_functions = activation_functions


class Model:
    """Model class"""

    @staticmethod
    def transform(model: any, framework: Framework):
        """Transform model to encoded form."""
        match framework:
            case Framework.KERAS:
                # Transformation
                return Model.get_keras_model_info(model)
            case _:
                raise NotImplementedError("Framework not supported!")

    @staticmethod
    def activation_to_str(activation) -> str:
        """Translate activation function to string"""
        match activation:
            case keras.activations.softmax:
                return "softmax"
            case keras.activations.relu:
                return "relu"
            case keras.activations.tanh:
                return "tanh"
            case keras.activations.sigmoid:
                return "sigmoid"
            case _:
                raise NotImplementedError("Activation function not supported.")

    # Return shape, weights, bias, activation functions from a keras model
    @staticmethod
    def get_keras_model_info(model: keras.Model) -> EncodedModel:
        """Encode model information."""
        layers = model.layers
        num_layers = len(layers)
        layer_shapes = []
        weights = []
        biases = []
        activation_functions = []

        for layer in layers:
            layer_shapes.append(layer.output_shape)

            # if keras don't use numpy in the future, we change this part
            weight = layer.get_weights()[0]
            bias = layer.get_weights()[1]
            weights.append(weight)
            biases.append(bias)

            activation_functions.append(Model.activation_to_str(layer.activation))

        return EncodedModel(num_layers, layer_shapes, weights, biases, activation_functions)
