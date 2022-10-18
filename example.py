from keras.models import Sequential
from keras.layers import Dense
import numpy as np

from ffnn import FFNN
from clustering import KMeansClusterer
from merging import GlobalMerger


shape = (4, 5, 6, 3)

model = Sequential([
    Dense(5, activation='relu', input_shape=(4,)),
    Dense(6, activation='relu'),
    Dense(3, activation='relu'),
])


weights = [layer.get_weights()[0] for layer in model.layers]
bias = [layer.get_weights()[1] for layer in model.layers]

# Todo: figure this out
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


activations = ["relu"] * 3
restored_model = FFNN(shape, weights, bias, activations)


# forward pass a [10 x 4] matrix, two data points with 4 features
np.random.seed(42)
dataset = np.random.rand(10, 4)
restored_model.forward_pass(dataset)

# cluster into 2 clusters
cluster_labels = KMeansClusterer.cluster(restored_model, 2)


# merge clusters
merged_model = GlobalMerger.merge(restored_model, cluster_labels)
restored_model.model.summary()
merged_model.model.summary()
