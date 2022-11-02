from abc import abstractmethod
from typing import Optional

import networkx as nx
from bokeh.io import output_file, show
from bokeh.plotting import from_networkx
from bokeh.models import (BoxZoomTool, HoverTool, Plot, ResetTool, PanTool, WheelZoomTool, MultiLine, Circle)
from bokeh.palettes import Spectral4

from ffnn import FFNN
from visualisation_graph import (Edge, Graph, Node, Layer)



def generate_nodes(mlp_idx_range: list[tuple[int]], edges: list[tuple[int, int, float]], attr_names: list[str]=None) \
                  -> tuple[dict, dict[str,Node], list[Edge], list[Layer]]:
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
                new_node = Node(n, x_pos, y_pos, name, {name: 1.0})
            else:
                new_node = Node(n, x_pos, y_pos, name)
            nodes.update({name: new_node})
            layer_nodes.append(new_node)
            edges = [(new_node, end, w) if start == n else (start, end, w) for start, end, w in edges]
            edges = [(start, new_node, w) if end == n else (start, end, w)for start, end, w in edges]
        layers.append(Layer(layer_nodes))

    vis_graph_edges = [Edge(start, end, w) for start, end, w in edges]
    for edge in vis_graph_edges:
        print(f"edge starting at {edge.start_node.feature_name} and ending at {edge.end_node.feature_name}")
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
        # TODO: edges
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
    def visualise(mlp: FFNN, features: Optional[list[str]] = None) -> None:
        """ Generate interactisve visualisation using networkx and bokeh.
        :params
            mlp: FFNN
                the network to be visualised
        """
        (G, graph, pos_nodes, _) = SimpleVisualizer._generate_networkx_model(mlp, features)

        ATTACK, SUPPORT = "red", "green"
        edge_colors = {}
        edge_weights = {}
        edge_type = {}
        supports = {}
        attacks = {}
        print(G.edges())
        for start_node_idx, end_node_idx, d in G.edges(data=True):
            edge_colors[(start_node_idx, end_node_idx)] = ATTACK if d['weight'] < 0 else SUPPORT
            edge_weights[(start_node_idx, end_node_idx)] = d['weight']
            edge_type[(start_node_idx, end_node_idx)] = "Attack" if d['weight'] < 0 else "Support"
            start_node = graph.get_node(start_node_idx)
            end_node = graph.get_node(end_node_idx)
            end_node.transfer_attack_support(start_node.supports, d['weight'])

        for (idx, node) in graph.nodes.items():
            print(f"support nodes for node {idx} are {node.get_supporting_nodes()}")
            supports.update({idx: ', '.join(node.get_supporting_nodes())})
            attacks.update({idx: ', '.join(node.get_attacking_nodes())})


        nx.set_node_attributes(G, supports, "supports")
        nx.set_node_attributes(G, attacks, "attacks")
        nx.set_edge_attributes(G, edge_colors, "edge_color")
        nx.set_edge_attributes(G, edge_weights, "edge_weight")
        nx.set_edge_attributes(G, edge_type, "edge_type")

        graph = from_networkx(G, pos_nodes)

        node_hover_tool = HoverTool(tooltips=[("index", "@index"), ("supports", "@supports"), ("attacks", "@attacks")],\
            renderers=[graph.node_renderer])
        edge_hover_tool = HoverTool(tooltips=[("edge_weight", "@edge_weight"), ("edge_type", "@edge_type")],
                                    renderers=[graph.edge_renderer], line_policy='interp')


        # plot = Plot(width=WIDTH, height=HEIGHT, x_range=Range1d(-1.1, 1.1),
        #             y_range=Range1d(-1.1, 1.1), sizing_mode='scale_both')
        plot = Plot(sizing_mode='scale_both')


        plot.add_tools(node_hover_tool, edge_hover_tool, BoxZoomTool(), ResetTool(), PanTool(), WheelZoomTool())

        graph.edge_renderer.glyph = MultiLine(line_color="edge_color", line_alpha=0.8, line_width="edge_weight")
        graph.node_renderer.glyph = Circle(size=35, fill_color=Spectral4[0])
        plot.renderers.append(graph)

        output_file("networkx_graph.html")
        show(plot, sizing_mode='stretch_both')
