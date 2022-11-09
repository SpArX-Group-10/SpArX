from enum import Enum
import jsonpickle

class EdgeType(Enum):
    """Specifies the edge type after clustering."""
    ATTACK = 'red'
    SUPPORT = 'green'


class Node:
    """The building block of a neural network."""

    def __init__(self, idx: int, x: float, y: float, feature_name,
                 incoming=None):  # pylint: disable=dangerous-default-value
        """Initialize node."""
        if incoming is None:
            incoming = {}
        self.idx = idx
        self.x = x
        self.y = y
        self.feature_name = feature_name
        self.incoming = incoming

    def rename(self, name: str):
        """Rename feature."""
        self.feature_name = name

    def transfer_attack_support(self, supports: dict[str, float], weight: float):
        """Propagate attacks and supports."""
        for n, v in supports.items():
            self.incoming.update({n: self.incoming.get(n, 0.0) + v * weight})

    def get_support_attack_nodes(self) -> tuple[list[str], list[str]]:
        """Get support nodes."""
        print(self.incoming)
        supports = [label for label, w in self.incoming.items() if w > 0]
        attacks = [label for label, w in self.incoming.items() if w < 0]
        return supports, attacks

    def add_incoming_node(self, label: str, weight: float):
        """Adds a node which supports or attacks the current node"""
        self.incoming.update({label: weight})

    def __repr__(self) -> str:
        return self.feature_name

    def toJSON(self):
        """This method serializes a Python Object to JSON"""
        return jsonpickle.encode(self)


class Edge:
    """Connects one node to the other between two adjacent layers."""

    def __init__(self, start_node: Node, end_node: Node, weight: float):
        self.start_node = start_node
        self.end_node = end_node
        self.weight = weight
        self.edge_type = EdgeType.ATTACK if weight < 0 else EdgeType.SUPPORT

    def __repr__(self) -> str:
        return f"Edge from {self.start_node} to {self.end_node} with weight {self.weight:.2f} \n"

    def toJSON(self):
        """This method serializes a Python Object to JSON"""
        return jsonpickle.encode(self)


class Layer:
    """Contains a series of nodes in the neural network."""

    def __init__(self, nodes: list[Node]):
        self.nodes = nodes
        self.num_nodes = len(nodes)

    def toJSON(self):
        """This method serializes a Python Object to JSON"""
        return jsonpickle.encode(self)


class Graph:
    """Neural network in matrix form."""

    def __init__(self, nodes: dict[str, Node], edges: list[Edge]):
        """Initialize graph."""
        self.nodes = nodes
        self.edges = edges

    def get_node(self, name: str) -> Node:
        """Get node."""
        return self.nodes.get(name)

    def toJSON(self):
        """This method serializes a Python Object to JSON"""
        return jsonpickle.encode(self)
