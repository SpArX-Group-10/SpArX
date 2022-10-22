from enum import Enum, auto
import re

# Model Creation
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPool1D, Input, concatenate, Dropout, Activation
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import DataFrame
from typing import Optional, Tuple

import legacy.compas_load_and_preprocess

# hidden layers
HIDDEN_LAYERS = [50, 50]
# Training parameters
EPOCHS = 1000
PATIENCE = 30
BATCH_SIZE = 64

## Added code 
class Framework(Enum):
    KERAS = auto()

# Approach 1: User inputs a pre-trained model
def import_model(framework: Framework, model: any) -> keras.Model:
    match framework:
        case Framework.KERAS:
            if not model.isinstance(keras.Model):
                raise ValueError("Model is not a Keras model.")
            if not verify_keras_model_is_fnn(model): 
                raise ValueError("Model is not a feed-forward neural network.")
            return model
        case default:
            raise ValueError("Unsupported framework: {}!", framework)

def verify_keras_model_is_fnn(model: keras.Model) -> bool:
    # TODO: verify that the model is a feed-forward neural network
    # Verify all layers are dense layers
    for layer in model.layers:
        if not layer.isinstance(keras.layers.Dense):
            return False
    # Verify that the model is a sequential model
    return model.isinstance(keras.Sequential)
    
# Approach 2: we train it using
# - Dataset
# - number of layers for MLP
# - number of hidden neurons for each hidden layer
# - activation functions for MLP

def train_model(dataset: str, activation_functions: list[str], hidden_layers_size: list[int]):
    X, y = import_dataset(dataset)
    model = get_FFNN_model_general(X, y, activation_functions, hidden_layers_size)
    
    # divide test and train (one-hot and original format)
    # TODO: get user information for splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=2, shuffle=True)
    net_train(model, X_train, y_train, X_test, y_test)
    return model

# Import dataset from file path to pandas dataframe
def import_dataset(filepath: str, features: Optional[list[str]]=None) -> Tuple[DataFrame, DataFrame]:
    # Assuming the dataset is in the same directory as the module
    # Assuming last column is the label and the rest are features
    # Assuming first row is the header
    raw_data = pd.read_csv(filepath)
    # print(raw_data)
    if features:
        header = list(raw_data.columns[1:-1])
        both = set(features).intersection(header)
        feature_indeces = [header.index(x) for x in both]
        data_entries = raw_data.iloc[:, feature_indeces]
    else:
        data_entries = raw_data.iloc[:, 1:-1] # all rows, all columns except the last
        
    labels = raw_data.iloc[:, -1] # all rows, last column 
    return (data_entries, labels)


def load_preset_dataset(dataset: str) -> Tuple[DataFrame, DataFrame]:
    # Load and plot
    match dataset:
        case "breast cancer":
            data = load_breast_cancer()
            X = pd.DataFrame(data.data)
            y = pd.DataFrame(data.target)
            return (X, y)

        case "compass":
            # Load and plot
            data = compas_load_and_preprocess.load_compas()

            # ploly_df(data)
            CLASS = 'two_year_recid'

            # Split X and y
            X = data.drop(columns=[CLASS])
            y = data[CLASS]

            # Randomize
            X = X.sample(frac=1, random_state=2020)
            y = y.loc[X.index.values]
            X.reset_index(inplace=True, drop=True)
            y.reset_index(inplace=True, drop=True)

            return (X, y)
        
        case _:
            raise Exception("Unsupported dataset option.")




# evaluating model
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# constructing model. Structure: input, multiple hidden layers(relu), output(relu, sigmoid)
def get_FFNN_model(X, y, hidden_layers_size=[4]):
    """
        BASIC MODEL for the FF-NN
    """
    input_size = len(X.columns.values)
    output_size = len(y.columns.values)

    if len(hidden_layers_size) == 0:
        # No hidden layer (linear regression equivalent)
        ff_layers = [Dense(output_size, input_shape=(input_size,), activation='softmax')]
    else:
        # With sigmoid hidden layers
        ff_layers = [
            Dense(hidden_layers_size[0], input_shape=(input_size,), activation="relu"),
            Dense(output_size, activation='sigmoid')
        ]
        for hidden_size in hidden_layers_size[1:]:
            ff_layers.insert(-1, Dense(hidden_size, activation='relu'))

    print(ff_layers)
    model = Sequential(ff_layers)
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', recall_m, precision_m])
    model.summary()
    return model


# constructing model. Structure: input, multiple hidden layers, output
def get_FFNN_model_general(X: DataFrame, y: DataFrame, activation_funcs: list[str], hidden_layers_size: list[int]) -> Model:
    """
        BASIC MODEL for the FF-NN
    """
    input_size = len(X.columns.values)
    output_size = len(y.columns.values)

    if len(hidden_layers_size) == 0:
        # No hidden layer (linear regression equivalent)
        ff_layers = [Dense(output_size, input_shape=(input_size,), activation='softmax')]

    else:
        # With activation functions provided hidden layers
        ff_layers = [
            Dense(hidden_layers_size[0], input_shape=(input_size,), activation="relu"),
            Dense(output_size, activation='sigmoid')
        ]
        for (i, hidden_size) in enumerate(hidden_layers_size[1:]):
            ff_layers.insert(-1, Dense(hidden_size, activation=activation_funcs[i]))

    model = Sequential(ff_layers)
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', recall_m, precision_m])
    model.summary()
    return model



    # train FFNN
    def net_train(model, X_train, y_train_onehot, X_validate, y_validate_onehot, epochs=EPOCHS):

        # Train the model
        history = model.fit(X_train, y_train_onehot, verbose=2, epochs=epochs, batch_size=BATCH_SIZE,
                            validation_data=(X_validate, y_validate_onehot))

        return history
