import numpy as np
import tensorflow as tf
from keras import backend as K


class FFNN:
    """ feedforward neural network """

    def __init__(self, shape: tuple, weights: np.ndarray, bias: np.ndarray, activation_functions: list[str]):
        """Initialises a feed-forward neural network with the given weights, bias and shapes.

        :param
            shape: tuple
                The shape of the network.
            weights: np.ndarray
                The weights of the network.
            bias: np.ndarray
                The bias of the network.
            activation_functions: np.ndarray
                The activation functions of the network.
        """

        self.shape = shape
        self.weights = weights
        self.bias = bias
        self.activation_functions = activation_functions

        self.model = self._to_keras_model()
        self.functors = self._create_functors()


    def _to_keras_model(self) -> tf.keras.Model:
        """ Converts the current model to an keras model. """

        inputs = tf.keras.Input(shape=(self.shape[0],))
        layers = []
        for i in range(1, len(self.shape)):
            layer = tf.keras.layers.Dense(
                self.shape[i], activation=self.activation_functions[i-1])
            layer.set_weights(weights=self.weights[i] + self.bias[i])
            layers.append(layer)

        model = tf.keras.Sequential([inputs] + layers)
        return model


    def _create_functors(self) -> list[K.function]:
        """ Creates the activation functions for the network
        """
        inp = self.model.input  # input placeholder
        outputs = [layer.output for layer in self.model.layers]  # all layer outputs
        functors = K.function([inp], outputs)  # evaluation functions

        return functors


    def forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        """ Performs a forward pass through the network and return the activated output for each layer

        :param
            inputs: np.ndarray
                The inputs to the network.
        """

        return self.functors([inputs])
