from benchmark import BenchmarkData, create_benchmark_data
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import pytest
from ffnn import FFNN
from clustering import KMeansClusterer
from merging import GlobalMerger

@pytest.fixture(scope = "session")
def benchmark_data() -> BenchmarkData:
    """ Returns the benchmark data run on our custom dataset from the legacy code. """
    return create_benchmark_data()

@pytest.fixture(scope = "session")
def model() -> Sequential:
    """ Returns Sequential Model. """
    return Sequential([
        Dense(6, activation='relu', input_shape=(4,), kernel_initializer=tf.keras.initializers.GlorotUniform(seed=5)),
        Dense(6, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=5)),
        Dense(3, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=5)),
    ])

@pytest.fixture(scope = "session")
def compiled_model(model: Sequential) -> FFNN:
    """ Returns Compiled Model. """
    shape = (4, 6, 6, 3)
    weights = [layer.get_weights()[0] for layer in model.layers]
    bias = [layer.get_weights()[1] for layer in model.layers]
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    activations = ["relu"] * 3
    return FFNN(shape, weights, bias, activations)

@pytest.fixture(scope = "session")
def forward_passed(compiled_model: FFNN) -> FFNN:
    """ Does a forward pass with the dataset. """
    dataset = np.loadtxt("./test/data/iris-test.data", delimiter=",", usecols=(0,1,2,3))
    compiled_model.forward_pass(dataset)
    return compiled_model

@pytest.fixture(scope = "session")
def clusters(forward_passed: FFNN) -> list[np.ndarray, np.ndarray]:
    """ Returns the clustering labels after clustering our custom dataset. """
    shrinkage = 0.5
    return KMeansClusterer.cluster(forward_passed, shrinkage, seed=1)

@pytest.fixture(scope = "session")
def merged_model(compiled_model: FFNN, clusters: list[np.ndarray, np.ndarray]):
    """ Returns Merged Model. """
    return GlobalMerger.merge(compiled_model, clusters)

# def test_compiled_model(compiled_model: FFNN, )


def test_clusters(benchmark_data: BenchmarkData, clusters: list[np.ndarray, np.ndarray]) -> None:
    """ Tests whether our clustering functions return the same as the legacy code. """
    assert all(np.array_equal(a,b) for a,b in zip(benchmark_data.clustering_labels, clusters))

def test_merged_model(benchmark_data: BenchmarkData, merged_model: FFNN) -> None:
    """ Tests that weights and biases of merged models match. """
    weights, biases, _, _ = benchmark_data.merged_model
    assert all(
        np.array_equal(b_layer_weight, layer.get_weights()[0])
        for b_layer_weight,layer in zip(weights, merged_model.model.layers))
    assert all(
        np.array_equal(b_layer_bias, layer.get_weights()[1])
        for b_layer_bias,layer in zip(biases, merged_model.model.layers))
