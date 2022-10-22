import unittest
from keras.models import Sequential # pylint: disable=import-error
from keras.layers import Activation, Dense, Input # pylint: disable=import-error
from module1 import import_dataset, import_model, Framework # pylint: disable=import-error

class Module1Test(unittest.TestCase):
    """ Tests. """

    # Testing approach 1: importing a model
    def test_import_model_keras_and_fnn(self):
        """ Import a Keras FFNN model. """
        ff_layers = [
            Dense(10, activation='relu'),
            Dense(2, activation='softmax')
        ]
        model = Sequential(ff_layers)
        self.assertEqual(import_model(Framework.KERAS, model), model)

    def test_import_unsupported_model(self):
        """ Fails when importing unsupported models. """
        model = "Not Keras Model!"
        self.assertRaises(ValueError, import_model, Framework.KERAS, model)

    def test_import_model_keras_and_not_fnn(self):
        """ Fails when importing a Keras model that is not FFNN. """
        ff_layers = [
            Input(shape=(10,)),
            Dense(10, activation='relu'),
            Dense(2, activation='softmax'),
            Activation('relu')
        ]
        model = Sequential(ff_layers)
        self.assertRaises(ValueError, import_model, Framework.KERAS, model)

    # Tests for approach 2: training a model given parameters
    def test_import_dataset(self):
        """ Imports raw dataset and process the data. """
        filepath = "tests/data/test_data.csv"
        data_entries, labels = import_dataset(filepath)
        self.assertEqual(data_entries.shape, (3, 2))
        self.assertEqual(labels.shape, (3,))

    def test_import_dataset_features(self):
        """ Imports raw dataset with selected features. """
        filepath = "tests/data/test_data.csv"
        data_entries, labels = import_dataset(filepath, ["name"])
        self.assertEqual(data_entries.shape, (3, 1))
        self.assertEqual(labels.shape, (3,))

    # def test_train_model(self):
    #     self.assertTrue(False and "not implemented")

    # def test_recall_m(self):
    #     self.assertTrue(False and "not implemented")

    # def test_precision_m(self):
    #     self.assertTrue(False and "not implemented")

    # def test_get_ffnn_model(self):
    #     self.assertTrue(False and "not implemented")

    # def test_get_general_ffnn_model(self):
    #     self.assertTrue(False and "not implemented")

    # def test_net_train(self):
    #     self.assertTrue(False and "not implemented")


if __name__ == '__main__':
    unittest.main()
