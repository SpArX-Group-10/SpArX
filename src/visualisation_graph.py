from enum import Enum

class EdgeType(Enum):
    """Specifies the edge type after clustering."""
    ATTACK = 'red'
    SUPPORT = 'green'

class Node:
    """The building block of a neural network."""
    def __init__(self, idx: int, x: float, y: float, feature_name: str):
        self.idx = idx
        self.x = x
        self.y = y
        self.feature_name = feature_name
        self.supports = []
        self.attacks = []

class Edge:
    """Connects one node to the other between two adjacent layers."""
    def __init__(self, start_node: Node, end_node: Node, weight: float):
        self.start_node = start_node
        self.end_node = end_node
        self.weight = weight
        self.edge_type = EdgeType.ATTACK if weight < 0 else EdgeType.SUPPORT


class Layer:
    """Contains a series of nodes in the neural network."""
    def __init__(self, nodes: list[Node]):
        self.nodes = nodes
        self.num_nodes = len(nodes)


class Graph:
    """Neural network in matrix form."""
    def __init__(self, layers: list[Layer]):
        self.layers = layers
