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

        self.activation_functions = activation_functions

        self.model = self._to_keras_model(shape, weights, bias, activation_functions)
        self.functors = self._create_functors()

        self.data = None

        # activation values for each layer
        self.forward_pass_data = None


    def _to_keras_model(
        self, shape: tuple[int], weights: np.ndarray, biases: np.ndarray, activations: str
    ) -> tf.keras.Model:
        """ Converts the current model to an keras model. """

        inputs = tf.keras.layers.InputLayer(input_shape=(shape[0],))
        layers = []
        # create the model shape and layers
        for i in range(1, len(shape)):
            layer = tf.keras.layers.Dense(shape[i], activation=activations[i-1])
            layers.append(layer)

        model = tf.keras.Sequential([inputs] + layers)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # set weights and biases for the model
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            model.layers[i].set_weights([weight, bias])

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

        self.data = inputs
        self.forward_pass_data = self.functors(inputs)

        return self.forward_pass_data


    def add_layer(self, neuron_count : int, weights: np.ndarray, bias: np.ndarray, activation_function: str) -> None:
        """ Extends the network with a new layer. """

        # add the new activaiton function
        self.activation_functions.append(activation_function)

        # update the inner keras model
        self.model.add(tf.keras.layers.Dense(neuron_count, activation=activation_function))
        self.model.layers[-1].set_weights([weights, bias])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # update the functors
        self.functors = self._create_functors()
        self.forward_pass(self.data)


    def get_shape(self) -> tuple[int]:
        """ Returns the shape of the network. """
        return (self.model.layers[0].input_shape[1], ) + tuple(layers.output_shape[1] for layers in self.model.layers)
