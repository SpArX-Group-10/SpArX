from abc import abstractmethod
from typing import Optional
import networkx as nx

from .ffnn import FFNN
from .visualisation_graph import (Edge, Graph, Node, Layer)

def generate_nodes(mlp_idx_range: list[tuple[int]], edges: list[tuple[int, int, float]], attr_names: list[str]=None) \
                  -> tuple[dict, dict[str, Node], list[Edge], list[Layer]]:
    """ Generate positions and nodes for visualisation """
    pos_nodes = {}
    nodes = {}
    layers = []
    SCALING_FACTOR = 5
    for layer, (start, end) in enumerate(mlp_idx_range):
        num_nodes = end - start + 1
        layer_nodes = []
        for i, n in enumerate(range(start, end)):
            x_pos = layer
            y_pos = (1 / num_nodes) * (i + 1) * SCALING_FACTOR
            pos_nodes.update({n : (x_pos, y_pos)})
            name = attr_names.get(n)
            new_node = None
            if layer == 0:
                new_node = Node(n, x=x_pos, y=y_pos, label=name, layer=0, incoming={name: 1.0})
            else:
                new_node = Node(n, x=x_pos, y=y_pos, label=name, layer=layer)
            nodes.update({name: new_node})
            layer_nodes.append(new_node)
            edges = [(new_node, end, w) if start == n else (start, end, w) for start, end, w in edges]
            edges = [(start, new_node, w) if end == n else (start, end, w)for start, end, w in edges]
        layers.append(Layer(layer_nodes))

    vis_graph_edges = [Edge(start, end, w) for start, end, w in edges]

    return pos_nodes, nodes, vis_graph_edges, layers

def generate_weights(mlp: FFNN, G: nx.DiGraph, mlp_idx_range: list[tuple[int]]) -> list[tuple[int, int, float]]:
    """ Generate weights for visualisation """
    edges = []
    for l_idx, layer in enumerate(mlp.model.layers):
        weights = layer.get_weights()
        for out_idx, _ in enumerate(weights[0]):
            for in_idx, weight in enumerate(weights[0][out_idx]):
                cur_layer_node = mlp_idx_range[l_idx][0] + out_idx
                next_layer_node = mlp_idx_range[l_idx + 1][0] + in_idx
                edges.append((cur_layer_node, next_layer_node, weight))
                G.add_edge(cur_layer_node, next_layer_node, color='r', weight=weight)

    return edges

def relabel_nodes(mlp_idx_range: list[tuple[int]], attr_names: list[str]) -> dict:
    """ Relabel nodes given the feature names, layer type and index """
    new_labels = {}
    for layer, (start, end) in enumerate(mlp_idx_range):
        if layer == 0:
            new_labels.update({lnode_idx: attr_names[lnode_idx] for lnode_idx in range(len(attr_names))})
        elif layer == len(mlp_idx_range) - 1:
            new_labels.update({start + out_idx: f"O{out_idx}" for out_idx in range(end - start + 1)})
        else:
            new_labels.update({start + lnode_idx: f"C{start + lnode_idx - mlp_idx_range[0][1]}" for \
                lnode_idx in range(end - start + 1)})
    return new_labels

def get_attack_support_by_node(graph: Graph, propagate: bool=False) -> tuple[dict[str, list[str]],
                                                                             dict[str, list[str]]]:
    """Constructs dictionary mapping from nodes labels to attack and support labels"""
    supports = {}
    attacks = {}
    for edge in graph.edges:
        start = edge.start_node
        end = edge.end_node
        # if true propagate attacks and supports among layers
        if propagate:
            end.transfer_attack_support(start.incoming, edge.weight)
        else:
            end.add_incoming_node(start.label, edge.weight)

    for label, node in graph.nodes.items():
        curr_support, curr_attack = node.get_support_attack_nodes()
        supports.update({label: curr_support})
        attacks.update({label: curr_attack})
    return supports, attacks


class Visualiser:
    """ Base visualisation class """
    @staticmethod
    @abstractmethod
    def visualise(mlp: FFNN) -> None:
        """ visualises the given mlp as a qbaf

        :params
            mlp: FFNN
                the network to be visualised
        """
        raise NotImplementedError

class SimpleVisualizer(Visualiser):
    """ Generates a basic representation of the clustered model using networkx """
    @staticmethod
    def _generate_networkx_model(mlp: FFNN, attr_names: list[str]=None) \
                                -> tuple[nx.DiGraph, Graph, dict, list[Layer]]:
        G = nx.DiGraph()

        mlp_shapes = mlp.get_shape()
        mlp_idx_range = []

        # calculate the offset in a list from start to end
        offset = 0
        for shape in mlp_shapes:
            mlp_idx_range.append((offset, offset + shape))
            offset = offset + shape

        if attr_names is None:
            attr_names = [f"X{idx}" for idx in range(mlp_shapes[0])]
        else:
            assert len(attr_names) == mlp_shapes[0]

        # loop through the weights of each layer and add them to the graph
        edges = generate_weights(mlp, G, mlp_idx_range)

        # relabel the nodes with corresponding arguments
        new_labels = relabel_nodes(mlp_idx_range, attr_names)

        # calculate the position of each node, along with the custom object edges and layers
        pos_nodes, nodes, vis_graph_edges, layers = generate_nodes(mlp_idx_range, edges, new_labels)

        graph = Graph(nodes, vis_graph_edges)

        G = nx.relabel_nodes(G, new_labels)
        new_pos_nodes = {new_labels[node]: pos for (node, pos) in pos_nodes.items()}

        return (G, graph, new_pos_nodes, layers)

    @staticmethod
    def visualise(mlp: FFNN, features: Optional[list[str]] = None) -> str:
        """ Generate interactive visualisation using networkx and bokeh.
        :params
            mlp: FFNN
                the network to be visualised
        """

        # TODO: add layer as return value if needed
        (_, custom_graph, _, _) = SimpleVisualizer._generate_networkx_model(mlp, features)
        get_attack_support_by_node(custom_graph)

        # FOR COMM
        return custom_graph.toJSON()
