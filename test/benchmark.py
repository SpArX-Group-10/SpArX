from test.legacy_test import legacy_benchmark_iris
import numpy as np

class BenchmarkData:
    """Class for storing results from test run on the legacy code."""

    def __init__(self, clustering_labels: np.ndarray) -> None:
        self.clustering_labels = clustering_labels


def create_benchmark_data() -> BenchmarkData:
    """Uses the legacy code to create the Benchmark Data."""

    return legacy_benchmark_iris.get_benchmarks()
