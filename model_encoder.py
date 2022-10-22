from enum import Enum, auto

from tensorflow import keras
from typing import Optional, Union

class Framework(Enum):
    KERAS = auto()
    PYTORCH = auto()

"""
We assume that model is already in-memory (or loaded if saved)
and passed in directly as the 
"""

class EncodedModel:
    def __init__(self, num_layers, layer_shapes, weights, biases, activation_functions):
        self.num_layers = num_layers
        self.layer_shapes = layer_shapes
        self.weights = weights
        self.biases = biases
        self.activation_functions = activation_functions

class Model:
    @staticmethod
    def transform(model: any, framework: Framework):
        match framework:
            case Framework.KERAS:
            # Transformation
                return Model.get_keras_model_info(model)
            case _:
                raise NotImplementedError("Framework not supported!")

    @staticmethod
    def activation_to_str(activation) -> str:
        # sigmoid, leaky relu (not implemented), elu, 
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
        layers = model.layers
        num_layers = len(layers)
        layer_shapes = []
        weights = []
        biases = []
        activation_functions = []

        for layer in layers:
            layer_shapes.append(layer.output_shape)

            # if keras don't use numpy in the future, we change this part
            w = layer.get_weights()[0]
            b = layer.get_weights()[1]
            weights.append(w)
            biases.append(b)

            activation_functions.append(Model.activation_to_str(layer.activation))

        return EncodedModel(num_layers, layer_shapes, weights, biases, activation_functions)


"""
keras.sequential([
    dense(64, activaiton="relu"),
    dense(64, activaiton="relu"),
    dense(64, activaiton="relu"),
    dense(64, activaiton="relu"),
    dense(64, activaiton="relu"),
    dense(64, activaiton="relu"),
    dense(3, activaiton="sigmoid"),
])

keras.fit(xtrain, xtest)
"""