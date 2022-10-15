from enum import Enum, auto

from tensorflow import keras
from typing import Optional, Union

class Framework(Enum):
    KERAS = auto()
    # PYTORCH = auto()

"""
We assume that model is already in-memory (or loaded if saved)
and passed in directly as the 
"""

class Model:
    @staticmethod
    def transform(model: any, framework: Framework):
        match framework:
            case Framework.KERAS:
            # Validation
                if not isinstance(model, keras.Model):
                    raise ValueError("Model is not a Keras model!")

                if not Model.is_keras_model_mlp(model):
                    raise ValueError("Model does not conform to MLP!")

            # Transformation
                return Model.get_keras_model_info(model)
            case _:
                raise NotImplementedError("Framework not supported!")
    
    
    """
        TODO: handle custom objects (model, layer, etc)
        Check that model is type Sequential, and layer is of type Dense
        return isinstance(model, keras.Model.Sequential) and all((isinstance(layer, keras.layers.Dense) for layer in model.layers))

        validate its an mlp:
        (to research)
        - check layers are dense???
        - right now all layers seem to be dense layers
    """ 
    @staticmethod
    def is_keras_model_mlp(model: keras.Model) -> bool:
        for layer in model.layers:
            if not isinstance(layer, keras.layers.Dense):
                return False
        return True


    @staticmethod
    def activation_to_str(activation) -> str:
        # sigmoid, leaky relu (not implemented), elu, 
        if isinstance(activation, keras.activations.softmax):
            return "softmax"
        elif isinstance(activation, keras.activations.relu):  
            return "relu" 
        elif isinstance(activation, keras.activations.tanh):  
            return "tanh"
        elif isinstance(activation, keras.activation.sigmoid):
            return "sigmoid"
        else:
            raise NotImplementedError("Activation function not supported.")  

    # Return shape, weights, bias, activation functions from a keras model
    @staticmethod
    def get_keras_model_info(model: keras.Model):
        layers = model.layers
        num_layers: int = len(layers)
        layer_shapes = []
        weights = []
        biases = []
        activation_functions = []

        for (i, layer) in enumerate(layers):
            layer_shapes.append(layer.output_shape)

            # if keras don't use numpy in the future, we change this part
            w = layer.get_weights()[0]
            b = layer.get_weights()[1]
            weights.append(w)
            biases.append(b)

            activation_functions.append(Model.activation_to_str(layer.activation))

        return (num_layers, layer_shapes, weights, biases, activation_functions)



# create_ffnn(shape, activations, )
# create_ffnn([3, 50, 50, 50, 50, 3], activations, )
# train_ffnn(epochs, lr, optimizers, model, train, test)


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