from model_encoder import Model, Framework
import unittest
import keras
from keras.models import Sequential
from keras.layers import Dense, Input

class ModelEncoderTest(unittest.TestCase):
    
    def test_get_keras_model_info(self):
        ff_layers = [
            Input(shape=(5,)),
            Dense(2, activation='relu'),
        ]
        model = Sequential(ff_layers)
        res = Model.get_keras_model_info(model)
        self.assertEqual(res.num_layers, 1)
        self.assertEqual(res.layer_shapes, [(None, 2)])
        self.assertEqual(len(res.weights), 1)
        self.assertEqual(res.weights[0].shape, (5, 2))
        self.assertEqual(len(res.biases), 1)
        self.assertEqual(res.biases[0].shape, (2,))
        self.assertEqual(res.activation_functions, ['relu'])
        
    def test_activation_to_str(self):
        self.assertEqual(Model.activation_to_str(keras.activations.softmax), "softmax")
        self.assertEqual(Model.activation_to_str(keras.activations.relu), "relu")
        self.assertEqual(Model.activation_to_str(keras.activations.tanh), "tanh")
        self.assertRaises(NotImplementedError, Model.activation_to_str, "Not an activation function")
        
    def test_transform(self):
        ff_layers = [
            Input(shape=(5,)),
            Dense(2, activation='relu'),
        ]
        model = Sequential(ff_layers)
        self.assertRaises(NotImplementedError, Model.transform, model, Framework.PYTORCH)


if __name__ == '__main__':
    unittest.main()
