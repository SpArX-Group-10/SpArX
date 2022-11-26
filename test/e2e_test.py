import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import pytest

from src import FFNN, KMeansClusterer
from .benchmark import BenchmarkData, create_benchmark_data


@pytest.fixture(scope="session")
def benchmark_data() -> BenchmarkData:
    """Returns the benchmark data run on our custom dataset from the legacy code."""
    return create_benchmark_data()


@pytest.fixture(scope="session")
def clusters() -> list[np.ndarray, np.ndarray]:
    """Returns the clustering labels after clustering our custom dataset."""
    shape = (4, 6, 6, 3)
    model = Sequential(
        [
            Dense(
                6, activation="relu", input_shape=(4,), kernel_initializer=tf.keras.initializers.GlorotUniform(seed=5)
            ),
            Dense(6, activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=5)),
            Dense(3, activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=5)),
        ]
    )

    weights = [layer.get_weights()[0] for layer in model.layers]
    bias = [layer.get_weights()[1] for layer in model.layers]
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    activations = ["relu"] * 3
    restored_model = FFNN(shape, weights, bias, activations)
    dataset = np.loadtxt("./test/data/iris-test.data", delimiter=",", usecols=(0, 1, 2, 3))
    restored_model.forward_pass(dataset)
    shrinkage = 0.5
    return KMeansClusterer.cluster(restored_model, shrinkage, seed=1)


def test_clusters(benchmark_data: BenchmarkData, clusters: list[np.ndarray, np.ndarray]) -> None:
    """Tests whether our clustering functions return the same as the legacy code."""
    assert all(np.array_equal(a, b) for a, b in zip(benchmark_data.clustering_labels, clusters))
