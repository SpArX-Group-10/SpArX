from enum import Enum
import json

SCALING_FACTOR = 50
class EdgeType(Enum):
    """Specifies the edge type after clustering."""
    ATTACK = 'red'
    SUPPORT = 'green'


class Node:
    """The building block of a neural network."""

    def __init__(self, idx: int, x: float, y: float, label: str, layer: int,
                 incoming=None):  # pylint: disable=dangerous-default-value
        """Initialize node."""
        if incoming is None:
            incoming = {}
        self.idx = idx
        self.x = x
        self.y = y
        self.label = label
        self.incoming = incoming
        self.layer = layer

    def rename(self, name: str):
        """Rename feature."""
        self.label = name

    def transfer_attack_support(self, supports: dict[str, float], weight: float):
        """Propagate attacks and supports."""
        for n, v in supports.items():
            self.incoming.update({n: self.incoming.get(n, 0.0) + v * weight})

    def get_support_attack_nodes(self) -> tuple[list[str], list[str]]:
        """Get support nodes."""
        self.incoming = dict(sorted(self.incoming.items(), key = lambda item: item[1]))
        supports = [label for label, w in self.incoming.items() if w > 0]
        attacks = [label for label, w in self.incoming.items() if w < 0]
        return supports, attacks

    def add_incoming_node(self, label: str, weight: float):
        """Adds a node which supports or attacks the current node"""
        self.incoming.update({label: weight})

    def __repr__(self) -> str:
        return self.label

    def toDict(self):
        """This method produces a JSON representation of the object."""

        supporting_nodes, attacking_nodes = self.get_support_attack_nodes()
        json_dict = {}
        json_dict["id"] = self.idx
        json_dict["position"] = {"x": round(self.x * SCALING_FACTOR, 1) , "y": round(self.y * SCALING_FACTOR, 1)}
        json_dict["layer"] = self.layer
        json_dict["label"] = self.label
        json_dict["incoming"] = {k: float(v) for k, v in self.incoming.items()}
        json_dict["supporting_nodes"] = supporting_nodes
        json_dict["attacking_nodes"] = attacking_nodes
        return json_dict


class Edge:
    """Connects one node to the other between two adjacent layers."""

    def __init__(self, start_node: Node, end_node: Node, weight: float):
        self.start_node = start_node
        self.end_node = end_node
        self.weight = weight
        self.layer = start_node.layer
        self.edge_type = EdgeType.ATTACK if weight < 0 else EdgeType.SUPPORT

    def __repr__(self) -> str:
        return f"Edge from {self.start_node} to {self.end_node} with weight {self.weight:.2f} \n"

    def toDict(self):
        """This method produces a JSON representation of the object."""

        json_dict = {}
        json_dict["start_node"] = self.start_node.idx
        json_dict["end_node"] = self.end_node.idx
        json_dict["layer"] = self.layer
        json_dict["weight"] = float(self.weight)
        json_dict["edge_type"] = self.edge_type.name
        return json_dict


class Layer:
    """Contains a series of nodes in the neural network."""

    def __init__(self, nodes: list[Node]):
        self.nodes = nodes
        self.num_nodes = len(nodes)



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
        node_arr = []
        edge_arr = []
        for (_, node) in self.nodes.items():
            node_arr.append(node.toDict())
        for edge in self.edges:
            edge_arr.append(edge.toDict())

        return json.dumps({"nodes": node_arr, "edges": edge_arr})
