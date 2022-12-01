# Model Creation
from typing import Optional
import keras
from keras import backend as kbackend
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import DataFrame
from .model_encoder import Framework

# hidden layers
HIDDEN_LAYERS = [50, 50]
# Training parameters
EPOCHS = 1000
PATIENCE = 30
BATCH_SIZE = 64

def import_model(framework: Framework, model: any) -> keras.Model:
    """Approach 1: User inputs a pre-trained model"""
    # Verification
    match framework:
        # Keras model
        case Framework.KERAS:
            if not isinstance(model, keras.Model):
                raise ValueError("Model is not a Keras model.")
            if not verify_keras_model_is_fnn(model):
                raise ValueError("Model is not a feed-forward neural network.")
            return model
        case _:
            raise ValueError("Unsupported framework!")


def verify_keras_model_is_fnn(model: keras.Model) -> bool:
    """Verify that the model is a feed-forward neural network."""
    # Check that all hidden layers are dense layers.
    for layer in model.layers:
        if not isinstance(layer, keras.layers.Dense):
            return False
    # Verify that the model is a sequential model.
    return isinstance(model, keras.Sequential)

# Approach 2: we train it using
# - Dataset
# - number of layers for MLP
# - number of hidden neurons for each hidden layer
# - activation functions for MLP


def train_model(
        dataset: str,
        activation_functions: list[str],
        hidden_layers_size: list[int],
        epochs: int = 10,
) -> keras.Model:
    """ Train model."""
    x_data, y_data = import_dataset(dataset)
    model = get_ffnn_model_general(x_data, y_data, activation_functions, hidden_layers_size)

    # divide test and train (one-hot and original format)
    # get user information for splitting data
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=.2, random_state=2, shuffle=True)
    net_train(model, x_train, y_train, x_test, y_test, epochs=epochs)
    return model


def import_dataset(
    filepath: str,
    features: Optional[list[str]] = None,
    has_index: bool = True,
) -> tuple[DataFrame, DataFrame]:
    """Import dataset from file path to pandas dataframe."""
    # Assuming the dataset is in the same directory as the module
    # Assuming last column is the label and the rest are features
    # Assuming first row is the header
    # Assume all features are numerical
    # Assume all labels are numerical (or string - but string support is not implemented)

    try:
        raw_data = pd.read_csv(filepath)
    except Exception as exc:
        raise FileNotFoundError("File not found.") from exc

    if features:
        header = list(raw_data.columns[1:-1])
        # Check if the features specified are within the dataset
        if not set(features).issubset(set(header)):
            raise ValueError("Feature(s) not found in dataset.")
        both = set(features).intersection(header)
        feature_indeces = [header.index(x) for x in both]
        data_entries = raw_data.iloc[:, feature_indeces]
    else:
        # skip the first column (index) and last column (label)
        if has_index:
            data_entries = raw_data.iloc[:, 1:-1]
        else:
            data_entries = raw_data.iloc[:, :-1]

    labels = raw_data.iloc[:, -1:]  # all rows, last column
    return (data_entries, labels)


def load_preset_dataset(dataset: str) -> tuple[DataFrame, DataFrame]:
    """Load and plot"""
    match dataset:
        case "breast cancer":
            data = load_breast_cancer()
            x_data = pd.DataFrame(data.data)  # pylint: disable=no-member
            y_data = pd.DataFrame(data.target)  # pylint: disable=no-member
            return (x_data, y_data)

        case _:
            raise Exception("Unsupported dataset option.")


def recall_m(y_true, y_pred):
    """ Recall function."""
    true_positives = kbackend.sum(kbackend.round(kbackend.clip(y_true * y_pred, 0, 1)))
    possible_positives = kbackend.sum(kbackend.round(kbackend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + kbackend.epsilon())
    return recall


def precision_m(y_true, y_pred):
    """ Precision function."""
    true_positives = kbackend.sum(kbackend.round(kbackend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = kbackend.sum(kbackend.round(kbackend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + kbackend.epsilon())
    return precision

# constructing model. Structure: input, multiple hidden layers(relu), output(relu, sigmoid)


def get_ffnn_model(x_data, y_data, hidden_layers_size=[4]):  # pylint: disable=dangerous-default-value
    """
        Legacy code for BASIC MODEL for the FF-NN
    """
    input_size = len(x_data.columns.values)
    output_size = len(y_data.columns.values)

    if len(hidden_layers_size) == 0:
        # No hidden layer (linear regression equivalent)
        ff_layers = [
            Dense(
                output_size,
                input_shape=(
                    input_size,
                ),
                activation='softmax')]
    else:
        # With sigmoid hidden layers
        ff_layers = [
            Dense(
                hidden_layers_size[0], input_shape=(
                    input_size,), activation="relu"), Dense(
                output_size, activation='sigmoid')]
        for hidden_size in hidden_layers_size[1:]:
            ff_layers.insert(-1, Dense(hidden_size, activation='relu'))

    model = Sequential(ff_layers)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', recall_m, precision_m])

    return model


# constructing model. Structure: input, multiple hidden layers, output
def get_ffnn_model_general(
        x_data: DataFrame,
        y_data: DataFrame,
        activation_funcs: list[str],
        hidden_layers_size: list[int],
        output_activaiton: Optional[str] = "sigmoid") -> Model:
    """
        BASIC MODEL for the FF-NN
    """
    input_size = len(x_data.columns.values)
    output_size = len(y_data.columns.values)

    if len(hidden_layers_size) == 0:
        # No hidden layer (linear regression equivalent)
        ff_layers = [Dense(output_size, input_shape=(input_size,), activation=output_activaiton)]

    else:
        # With activation functions provided hidden layers
        ff_layers = [
            Input(shape=(input_size,)),
            Dense(output_size, activation=output_activaiton)
        ]

        for (i, hidden_size) in enumerate(hidden_layers_size):
            ff_layers.insert(-1, Dense(hidden_size, activation=activation_funcs[i]))


    model = Sequential(ff_layers)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# train FFNN


def net_train(  # pylint: disable=too-many-arguments
        model,
        x_train,
        y_train_onehot,
        x_validate,
        y_validate_onehot,
        epochs=EPOCHS):
    """Train the model"""
    history = model.fit(
        x_train,
        y_train_onehot,
        verbose=2,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_data=(
            x_validate,
            y_validate_onehot))

    return history
