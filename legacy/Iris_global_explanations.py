#!/usr/bin/env python

"""
------------------------------------------
Global Clustering using all the inputs.
Global Explanations and Local Explanations.
------------------------------------------
This code includes the approach for shrinking a two layer FFNN to a new network.
The new model clusters the hidden nodes in the original network based on their activations.
-----------------steps to achieve a shrunken model---------------------
1) This means that using all the inputs, we first compute the activations of the hidden nodes.
2) Then, using all these activations, we cluster the nodes that topically have the same (or close)
 activations.
3) Then the weights are computed independent of the inputs.
* There are multiple choices for the incoming weights of the hidden nodes:
  a) Mean
  b) Weights by random selection of a node at each cluster.
* For the outgoing weights from the hidden node, we can use sum as a natural choice.
------------------------ Convert to QBAF ----------------------------
4) Afterwards, we convert the resulting network to a QBAF by interpreting negative weights as attack and
 positive weights as supports and we visualize it.
* In the visualization step, green edges show support and negative edges show attack.
* The width of each edge shows the strength of attack or support relation (based on their weights).

 """

# Importing all the libraries and utilities
import legacy.utility_functions as utility_functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from datetime import datetime, date, timedelta
import string
import pickle
import json
import sys
import os
import itertools
import random
import math
import time
import re
import hashlib
import warnings
import subprocess

from keras.utils import plot_model
import os
import copy
import utility_functions


from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Model Creation
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPool1D, Input, concatenate, Dropout, Activation

# Model Training
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.cluster import KMeans, AgglomerativeClustering

"""
------------------------------------------
Hyper Params + setup
------------------------------------------
"""

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 35)

# hidden layers
HIDDEN_LAYERS = [50, 50]

# How much should the network be shrunken
# Integers in range (0-100). how much do you want the network be shrinked?
# (in percentatge)
Shrinkage_percentage = 50
# How much of the edges with low weights should be pruned in visualization
# (notice that they are not really pruned in the computation process)?

# this parameter is only used for visualization step
pruning_ratio = 0.5
# the number should be in range [0, 1].
# Example: pruning_ratio = 0.8 means that only the edges higher than 0.8
# quantile of all the weights would be shown.

preserve_percentage = 100 - Shrinkage_percentage

EPOCHS = 10
PATIENCE = 20
BATCH_SIZE = 64

os.environ['PYTHONHASHSEED'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# setup code
# making sure gpu donm't run out of memeory
# seed random
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    import tensorflow as tf
    import keras
    # import keras.backend as K
    from tensorflow.python.keras import backend as K
    from keras.utils.np_utils import to_categorical

    config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    tf.compat.v1.set_random_seed(1)
    sess = tf.compat.v1.Session(
        graph=tf.compat.v1.get_default_graph(),
        config=config)
    K.set_session(sess)

    # Check that we are using the standard configuration for channels
    assert K.image_data_format() == 'channels_last'


"""
------------------------------------------
pre-processing
------------------------------------------
"""

RESULT_PATH = './results'

if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)


def load_iris():
    # class could be one of ['Iris-setosa', 'Iris-versicolor',
    # 'Iris-versicolor']
    columns_names = [
        "sepal length",
        "sepal width",
        "petal length",
        "petal width",
        "class"]
    df = pd.read_csv('data/iris.data', header=None, names=columns_names)
    return df


# Load and plot
data = load_iris()
X = data.drop(columns="class")
y = data[["class"]]

# Randomize
X = X.sample(frac=1, random_state=2020)
y = y.loc[X.index.values]
X.reset_index(inplace=True, drop=True)
y.reset_index(inplace=True, drop=True)

# # One-hot
y_onehot = pd.get_dummies(y)


X_train, X_test, y_train, y_test, data_train, data_test, y_onehot_train, y_onehot_test = \
    train_test_split(X, y, data, y_onehot, test_size=.2, random_state=2, shuffle=True)


# metric functions
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


# metric functions
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


"""
------------------------------------------
Model Creation + Training
------------------------------------------
"""


# Creates a neural network model
def get_FFNN_model(X, y, hidden_layers_size=[4]):
    """
        BASIC MODEL for the FF-NN
    """
    input_size = len(X.columns.values)
    output_size = len(y.columns.values)

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
                    input_size,), activation='relu'), Dense(
                output_size, activation='softmax')]
        for hidden_size in hidden_layers_size[1:]:
            ff_layers.insert(-1, Dense(hidden_size, activation='relu'))

    model = Sequential(ff_layers)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', recall_m, precision_m])

    model.summary()

    return model


# train the network
def net_train(
        model,
        bestmodel_path,
        X_train,
        y_train_onehot,
        X_validate,
        y_validate_onehot,
        epochs=EPOCHS):
    # Define four callbacks to use
    checkpointer = ModelCheckpoint(
        filepath=bestmodel_path,
        verbose=1,
        save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE)

    # Train the model
    history = model.fit(
        X_train,
        y_train_onehot,
        verbose=2,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[
            checkpointer,
            early_stopping],
        validation_data=(
            X_validate,
            y_validate_onehot))

    return history


model = get_FFNN_model(X, y_onehot, HIDDEN_LAYERS)

model_path = os.path.join(RESULT_PATH, 'global_net_iris.h5')
forge_gen = False

# Loads the model if already exists and trained, if not train it and save it
if not os.path.exists(model_path) or forge_gen:
    history = net_train(
        model,
        model_path,
        X_train,
        y_onehot_train,
        X_test,
        y_onehot_test)

    score = model.evaluate(X_test, y_onehot_test)
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


predictions = np.argmax(model.predict(X_test), axis=1)
y_pred = np.eye(np.max(predictions) + 1)[predictions]
# y_test_labels = np.argmax(y_onehot_test.values, axis=1)
print("Classification report for the original model.")
print(classification_report(y_onehot_test, y_pred, digits=4))

# TODO: change this
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
plot_model(
    model,
    to_file=RESULT_PATH +
    '/model.png',
    show_shapes=True,
    show_layer_names=False)


"""
------------------------------------------
Clustering methods
------------------------------------------
"""

# clusting each hidden layer based on its activation (global clustering)


def clustering_nodes(
        preserve_percentage,
        NUMBER_OF_NODES_IN_HIDDEN_LAYER,
        activations,
        clustering_algorithm="kmeans"):
    # Shrink the network using the Kmeans clustering below.
    # Input: the activations of all the layers of the network (original model) using all the data instances (examples)
    # Output: a nested list showing that at each layer  what would be the new cluster label of each node.
    #         For example if the original model has [4, 6] hidden nodes (this means that the first hidden layer
    #         has 4 nodes and the second hidden layer has 6 nodes) and
    #         the clustering algorithm produces [[0,1,0,1],[1,2,0,0,2,1]] list as output,
    #         then it means that the first and the third node in the first layer are assinged to the first cluster and
    #         the second and the fourth nodes are assigned to the second cluster.
    #         For the second layer the third and the fourth nodes are assigned to the first cluster (0)
    #         and the first and the last nodes are assigned to the second cluster (1)
    # and the second and the fifth are assigned to the third cluster (2).  .
    clustering_labels = []
    for index, hidden_layer in enumerate(NUMBER_OF_NODES_IN_HIDDEN_LAYER):
        activation = activations[index]
        clustering_input = activation.T
        # For global clustering (using all the examples), the number of clusters uses -1 because
        # we want to have a separate cluster of zeros in the local clustering phase. This way
        # we have all the zero activations for a specific example in cluster (0).
        # Therefore, we have -1 here and we add a cluster of zeros after calling
        # the
        # clustering_the_zero_activations_out_to_a_new_cluster_for_an_example
        # function
        n_clusters_ = int((preserve_percentage / 100) * hidden_layer)
        if clustering_algorithm == "kmeans":
            clustering = KMeans(
                n_clusters=n_clusters_,
                random_state=1).fit(clustering_input)
        elif clustering_algorithm == "AgglomerativeClustering":
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters_).fit(clustering_input)
        clustering_labels.append(clustering.labels_)
    return clustering_labels


def clustering_nodes_with_zeros_activations_as_one_cluster(
        preserve_percentage, HIDDEN_LAYERS, activation):
    mean_activation = np.mean(activation, axis=0)
    threshold = 0.1
    probably_not_zero_activations = np.where([mean_activation > threshold])[1]
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=int((preserve_percentage /
                                    100) *
                                   HIDDEN_LAYERS[0]) -
                    1, random_state=1).fit(activation[:, probably_not_zero_activations].T)
    labels_after_adding_cluster_of_zeros = []
    label_index = 0
    for idx, activation in enumerate(mean_activation):
        if activation <= threshold:
            labels_after_adding_cluster_of_zeros.append(0)
        else:
            labels_after_adding_cluster_of_zeros.append(
                kmeans.labels_[label_index] + 1)
            label_index += 1

    return np.array(labels_after_adding_cluster_of_zeros)


def turn_off_specific_node_and_see_output(
        model,
        input,
        node_layer,
        node_indices_list,
        node_index):
    new_model = keras.models.clone_model(model)
    new_model.set_weights(model.get_weights())
    node_indices_list = list(np.where(node_indices_list)[0])
    predictions = np.argmax(model.predict(input), axis=1)
    new_weights = copy.deepcopy(model.layers[node_layer + 1].get_weights())
    for index in node_indices_list:
        if node_indices_list[node_index] != index:
            new_weights[0][index] = 0
        else:
            new_weights[0][index] = np.sum(
                model.layers[node_layer + 1].get_weights()[0][node_indices_list])
    new_model.layers[node_layer + 1].set_weights(new_weights)
    new_predictions = np.argmax(new_model.predict(input), axis=1)
    changes_counter = 0
    for index, prediction in enumerate(list(predictions)):
        if prediction != new_predictions[index]:
            changes_counter += 1
    return changes_counter


def merge_nodes_globally(
        X_onehot,
        y_onehot,
        activations,
        model,
        shrunken_model,
        preserve_percentage,
        HIDDEN_LAYERS,
        clustering_labels):
    # Based on the clustering step, we now shrink the model.
    # The strategy is to keep the a node from each cluster in the
    # hidden layer (and average the connecting weights of the input to the hidden layer and the biases) and
    # in the outgoing weights layer compute the activations based on this
    # equation w* = (W.H)/h*.
    input_size = len(X_onehot.columns.values)
    output_size = len(y_onehot.columns.values)
    # weights of the final clustered FFNN
    weights = []
    # intermediate state of weights -- when hidden layer n has but clustered
    # but layer n + 1 has not been clustered yet
    outgoing_weights = [[model.layers[0].get_weights()[0]]]
    # [     [[6, 7, 8] , [3, 4, 5], [1, 2, 3]] # weights of that layer
    #       [4 , 5, 6] # weights of the bias
    # ]
    biases = []
    epsilon = 1e-30
    all_layer_sizes = [input_size]
    for hidden_layer in HIDDEN_LAYERS:
        all_layer_sizes.append(hidden_layer)
    all_layer_sizes.append(output_size)

    for index, hidden_layer in enumerate(HIDDEN_LAYERS):
        # create new dimension (i.e. index hidden layer)
        weights.append([])
        biases.append([])
        outgoing_weights.append([])
        # iterate through the cluster label space of the current hidden layer 
        for label in range(0, int((preserve_percentage / 100) * HIDDEN_LAYERS[index])):
            # average all the incoming weights from previous layer to nodes with the same cluster label in the current layer
            weights[index].append(np.mean(np.vstack(outgoing_weights[index]).T[clustering_labels[index] == label], axis=0))
            biases[index].append(np.mean(model.layers[index].get_weights()[1][clustering_labels[index] == label]))
            # create intermediate state of outgoing weights 
            # (i.e. state in which index + 1 layer has not been cluster but layer index has)
            # by summing all the outgoing weights from the current layer from nodes with the same label
            outgoing_weights[index + 1].append((np.sum(
                model.layers[index + 1].get_weights()[0][clustering_labels[index] == label], axis=0)).reshape((1, -1)))
    biases.append([model.layers[len(HIDDEN_LAYERS)].get_weights()[1]])
    weights.append(outgoing_weights[-1])
    # -1 to skip the last one which is alread3y in correct shape.
    for index in range(len(weights)):
        if index == len(weights) - 1:
            weights[index] = np.vstack(weights[index])
        else:
            weights[index] = np.vstack(weights[index]).T
        biases[index] = np.vstack(biases[index]).reshape(-1, )

    return weights, biases, input_size, output_size


# merge the nodes at each cluster and recompute the incoming and outgoing
# weights of edges
def merge_nodes(
        X_onehot,
        y_onehot,
        activations,
        model,
        preserve_percentage,
        HIDDEN_LAYERS,
        clustering_labels):
    # Based on the clustering step, we now shrink the model.
    # The strategy is to keep the a node from each cluster in the
    # hidden layer (and average the connecting weights of the input to the hidden layer and the biases) and
    # in the outgoing weights layer sum up the weights of all the nodes in the
    # cluster
    input_size = len(X_onehot.columns.values)
    output_size = len(y_onehot.columns.values)
    weights = []
    outgoing_weights = [[model.layers[0].get_weights()[0]]]
    biases = []
    for index, hidden_layer in enumerate(HIDDEN_LAYERS):
        weights.append([])
        biases.append([])
        outgoing_weights.append([])
        for label in range(
            0, int(
                (preserve_percentage / 100) * HIDDEN_LAYERS[index])):
            weights[index].append(np.mean(np.vstack(
                outgoing_weights[index]).T[clustering_labels[index] == label], axis=0))
            biases[index].append(np.mean(model.layers[index].get_weights()[
                                 1][clustering_labels[index] == label]))
            outgoing_weights[index + 1].append((np.sum(model.layers[index + 1].get_weights()[
                                               0][clustering_labels[index] == label], axis=0)).reshape((1, -1)))
    biases.append([model.layers[len(HIDDEN_LAYERS)].get_weights()[1]])
    weights.append(outgoing_weights[-1])
    # -1 to skip the last one which is already in correct shape.
    for index in range(len(weights)):
        if index == len(weights) - 1:
            weights[index] = np.vstack(weights[index])
        else:
            weights[index] = np.vstack(weights[index]).T
        biases[index] = np.vstack(biases[index]).reshape(-1,)

    return weights, biases, input_size, output_size


activations = utility_functions.compute_activations_for_each_layer(
    model, X_test.values)

overal_unfaithfulness_list = []
Shrinkage_percentage_list = []
for Shrinkage_percentage in np.arange(20, 90, 20):
    Shrinkage_percentage_list.append(Shrinkage_percentage / 100)
    preserve_percentage = 100 - Shrinkage_percentage

    clustering_labels = clustering_nodes(
        preserve_percentage, HIDDEN_LAYERS, activations)
    # print(clustering_labels)
    cluster_numbers = [np.max(clustering_label) +
                       1 for clustering_label in clustering_labels]
    truncated_model_dimensions = [int(
        (preserve_percentage / 100) * hidden_layer) for hidden_layer in HIDDEN_LAYERS]
    if cluster_numbers != truncated_model_dimensions:
        continue
    shrinked_model = utility_functions.get_FFNN_model_non_binary(
        X_test, y_onehot_test, truncated_model_dimensions)

    weights, biases, input_size, output_size = merge_nodes(
        X_test,
        y_onehot_test,
        activations,
        model,
        preserve_percentage,
        HIDDEN_LAYERS,
        clustering_labels)

    """
    ---------------------------
    Visualize the shrinked model
    ---------------------------
    """

    truncated_weights = []
    for index, weight in enumerate(weights):
        truncated_weights.append(weight)
        truncated_weights.append(biases[index])

    # shrinked_model.set_weights(
    #     [np.array(layer1_weighted).T, np.array(layer1_bias), np.array(layer2_weighted_sum), layer2_bias])
    shrinked_model.set_weights(truncated_weights)

    predictions_shrinked = np.argmax(shrinked_model.predict(X_test), axis=1)
    y_shrinked_pred = np.eye(output_size)[predictions_shrinked]

    predictions_shrinked_train = np.argmax(
        shrinked_model.predict(X_train), axis=1)
    y_shrinked_pred_train = np.eye(output_size)[predictions_shrinked_train]
    predictions_pred_train = np.argmax(model.predict(X_train), axis=1)
    y_pred_train = np.eye(output_size)[predictions_pred_train]
    print("Classification report for the shrunken model.")
    print(classification_report(y_onehot_test, y_shrinked_pred, digits=4))
    print("Now let's compare the original model and the shrunken model")
    print(classification_report(y_pred, y_shrinked_pred, digits=4))

    quantile_threshold = 0.5
    FASs = []

    # visualize the shrunken model as QBAF.
    for test_index in range(0, 20):  # range(len(np.array(X_onehot_test))):
        input = np.array(X_test)[test_index]
        output = np.array(y_onehot_test)[test_index]
        feature_names = X_test.columns.values

        number_of_hidden_nodes = [int(
            (preserve_percentage / 100) * hidden_layer) for hidden_layer in HIDDEN_LAYERS]

        all_weights = []
        for weight in weights:
            all_weights.extend(list(weight.reshape((-1,))))

        quantile = np.quantile(
            np.abs(
                np.array(all_weights)).reshape(
                1, -1), pruning_ratio)
        weight_threshold = quantile

        from plot_QBAF import general_method_for_visualize_attack_and_supports_QBAF

        general_method_for_visualize_attack_and_supports_QBAF(
            input,
            output,
            shrinked_model,
            feature_names,
            number_of_hidden_nodes,
            weight_threshold,
            weights,
            biases,
            Shrinkage_percentage,
            'iris_global_graphs(shrunken_model)',
            test_index)

    # visualize the original model
    for test_index in range(0, 20):  # range(len(np.array(X_onehot_test))):
        input = np.array(X_test)[test_index]
        output = np.array(y_onehot_test)[test_index]
        feature_names = X_test.columns.values
        # make a vector from all weights of the original network.
        all_weights_original = []
        original_weights = []
        for layer in model.layers:
            all_weights_original.extend(
                list(layer.get_weights()[0].reshape((-1,))))
            original_weights.append(layer.get_weights()[0])

        quantile = np.quantile(
            np.abs(
                np.array(all_weights_original)).reshape(
                1, -1), pruning_ratio)
        weight_threshold = quantile

        from legacy.plot_QBAF import visualize_attack_and_supports_QBAF, general_clustered_visualize_attack_and_supports_QBAF

        general_clustered_visualize_attack_and_supports_QBAF(
            input,
            output,
            model,
            feature_names,
            HIDDEN_LAYERS,
            weight_threshold,
            original_weights,
            biases,
            Shrinkage_percentage,
            'iris_global_graphs(original_model)',
            test_index,
            clustering_labels)

    css = "body {background: white;}"
    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-versicolor']

    overal_unfaithfulness = np.mean(
        np.sum(
            np.power(
                shrinked_model.predict(
                    X_test.values) -
                model.predict(
                    X_test.values),
                2),
            axis=1))
    print(
        f"Overal Unfaithfulness (ratio: {Shrinkage_percentage}): {overal_unfaithfulness }")
    overal_unfaithfulness_list.append(overal_unfaithfulness)

    fidelity = np.mean(
        np.sum(
            np.power(
                shrinked_model.predict(
                    X_test.values) -
                model.predict(
                    X_test.values),
                2),
            axis=1))
    print(f"Fidelity: {fidelity}")
    number_of_nodes = sum(truncated_model_dimensions) + y_onehot_test.shape[1]
    original_activations = utility_functions.compute_activations_for_each_layer(
        model, X_test.values)
    pruned_activations = utility_functions.compute_activations_for_each_layer(
        shrinked_model, X_test.values)
    structural_fidelity = 0
    for i, original_activation in enumerate(original_activations):
        pruned_activation = pruned_activations[i]
        if i != len(original_activations) - 1:
            for cluster_label in range(
                    int(HIDDEN_LAYERS[i] * preserve_percentage / 100)):
                if cluster_label in clustering_labels[i]:
                    structural_fidelity += np.sum(np.abs(
                        np.mean(original_activation[:, clustering_labels[i] == cluster_label], axis=1) - pruned_activation[
                            :, cluster_label]))
        else:
            structural_fidelity += np.sum(
                np.abs(pruned_activation - original_activation))

    structural_fidelity /= (number_of_nodes * X_test.values.shape[0])
    print(f"Structural fidelity = {structural_fidelity}")

print(Shrinkage_percentage_list)
print(overal_unfaithfulness_list)
