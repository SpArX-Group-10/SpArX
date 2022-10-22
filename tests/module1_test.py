from module1 import import_dataset, import_model, Framework
import unittest
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPool1D, Input, concatenate, Dropout, Activation

class Module1Test(unittest.TestCase):

    # Testing approach 1: importing a model
    def test_import_model_keras_and_fnn(self):
        ff_layers = [
            Dense(10, activation='relu'),
            Dense(2, activation='softmax')
        ]
        model = Sequential(ff_layers)
        self.assertEqual(import_model(Framework.KERAS, model), model)
    
    def test_import_unsupported_model(self):
        model = "Not Keras Model!"
        self.assertRaises(ValueError, import_model, Framework.KERAS, model)
        
    def test_import_model_keras_and_not_fnn(self):
        ff_layers = [
            Input(shape=(10,)),
            Dense(10, activation='relu'),
            Dense(2, activation='softmax')
        ]
        model = Sequential(ff_layers)
        self.assertRaises(ValueError, import_model, Framework.KERAS, model)

    # Tests for approach 2: training a model given parameters
    def test_import_dataset(self):
        filepath = "tests/data/test_data.csv"
        data_entries, labels = import_dataset(filepath)
        self.assertEqual(data_entries.shape, (3, 2))
        self.assertEqual(labels.shape, (3,))
        
    def test_import_dataset_features(self):
        filepath = "tests/data/test_data.csv"
        data_entries, labels = import_dataset(filepath, ["name"])
        self.assertEqual(data_entries.shape, (3, 1))
        self.assertEqual(labels.shape, (3,))
        

if __name__ == '__main__':
    Module1Test().test_import_dataset()
    Module1Test().test_import_dataset_features()
