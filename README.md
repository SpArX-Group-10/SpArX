# SpArX: Sparse Argumentative eXplanations for Neural Networks

![CI](https://github.com/SpArX-Group-10/SpArX/actions/workflows/pylint.yml/badge.svg)

## Packages:

The python version is 3.10.0.

# Getting started

1.  Clone repo from: [https://github.com/SpArX-Group-10/SpArX](https://github.com/SpArX-Group-10/SpArX)

    ```bash
    git clone https://github.com/SpArX-Group-10/SpArX
    ```

2.  Install requirements:

    ```bash
    pip install -r requirements.txt
    ```

# Example

```python
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

from sparx import FFNN, KMeansClusterer, LocalMerger, BokehVisualizer


# shrink to a decimal percentage
SHRINK_TO_PERCENTAGE = 0.5

shape = (4, 6, 6, 3)

model = Sequential([
    Dense(shape[1], activation='relu', input_shape=(shape[0],)),
    Dense(shape[2], activation='relu'),
    Dense(shape[3], activation='relu'),
])


weights = [layer.get_weights()[0] for layer in model.layers]
bias = [layer.get_weights()[1] for layer in model.layers]


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


activations = ["relu"] * 3
restored_model = FFNN(shape, weights, bias, activations)


# forward pass a [10 x 4] matrix, two data points with 4 features
np.random.seed(42)
dataset = np.random.rand(10, 4)
restored_model.forward_pass(dataset)


# cluster into 2 clusters
cluster_labels = KMeansClusterer.cluster(restored_model, SHRINK_TO_PERCENTAGE)

# merge clusters
merged_model = LocalMerger.merge(restored_model, cluster_labels)
restored_model.model.summary()
merged_model.model.summary()

# Bokeh Visualizer - to visualise neural networks locally
BokehVisualizer.visualise(merged_model)
```
