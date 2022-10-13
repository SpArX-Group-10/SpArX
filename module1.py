from enum import Enum

# Model Creation
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPool1D, Input, concatenate, Dropout, Activation
from sklearn.datasets import load_breast_cancer
import pandas as pd
from pd import DataFrame

import compas_load_and_preprocess

class Framework(Enum):
    KERAS = auto()
   
# Approach 1: we train it using
# - Dataset
# - number of layers for MLP
# - number of hidden neurons for each hidden layer
# - activation function for MLP

# Approach 2: Pretrained model
# - What attributes do team2 need?

## Added code 
def import_dataset(data):
    # TODO
    return None 

# Import from keras
def import_model(framework: Framework, filepath: str):
    match framework:
        case Framework.KERAS:
            return load_keras_model(filepath)
        case default:
            # TODO: throw an error 
            return None

def load_keras_model(filepath: str):
    model = keras.models.load_model(filepath) #TODO
    return None

    
def get_preset_dataset(dataset: str):
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


# constructing model. Structure: input, multiple hidden layers(relu), output(relu, sigmoid)
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
        # With sigmoid hidden layers
        ff_layers = [
            Dense(hidden_layers_size[0], input_shape=(input_size,), activation="relu"),
            Dense(output_size, activation='sigmoid')
        ]
        for (i, hidden_size) in enumerate(hidden_layers_size[1:]):
            ff_layers.insert(-1, Dense(hidden_size, activation=activation_funcs[i]))

    print(ff_layers)
    model = Sequential(ff_layers)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', recall_m, precision_m])
    model.summary()
    return model



# train FFNN
def net_train(model, bestmodel_path, X_train, y_train_onehot, X_validate, y_validate_onehot, epochs=EPOCHS):
    # Define four callbacks to use
    checkpointer = ModelCheckpoint(filepath=bestmodel_path, verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE)

    # Train the model
    history = model.fit(X_train, y_train_onehot, verbose=2, epochs=epochs, batch_size=BATCH_SIZE,
                        callbacks=[checkpointer, early_stopping], validation_data=(X_validate, y_validate_onehot))

    return history


model = get_FFNN_model(X_train, y_train, HIDDEN_LAYERS)

model_path = os.path.join(RESULT_PATH, 'cancer_global_net.h5')
forge_gen = False

if not os.path.exists(model_path) or forge_gen:
    history = net_train(model, model_path, X_train, y_train, X_test, y_test)

    score = model.evaluate(X_test, y_test)
    plt.figure(figsize=(14, 6))
    for key in history.history.keys():
        plt.plot(history.history[key], label=key)
    plt.legend(loc='best')
    plt.grid(alpha=.2)
    plt.title(f'batch_size = {BATCH_SIZE}, epochs = {EPOCHS}')
    plt.draw()
else:
    print('Model loaded.')
    model.load_weights(model_path)
