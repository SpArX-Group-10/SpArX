from abc import abstractmethod
from typing import Optional
from enum import Enum

import networkx as nx
from bokeh.io import output_file, show
from bokeh.plotting import from_networkx
from bokeh.models import (BoxZoomTool, HoverTool, Plot, ResetTool, PanTool, WheelZoomTool, MultiLine, Circle)
from bokeh.palettes import Spectral4
# from bokeh.models import Arrow, NormalHead

from ffnn import FFNN

class EdgeType(Enum): 
    ATTACK = 'red'
    SUPPORT = 'green'

class Node:
    def __init__(self, idx: int, x: float, y: float, supports: list[Edge], attacks:list[Edge], feature_name: str):
        self.idx = idx
        self.x = x
        self.y = y
        self.feature_name = feature_name
        self.supports = supports 
        self.attacks= attack
    

class Edge:
    def __init__(self, start_node: Node, end_node: Node, weight: float):
        self.start_node = start_node
        self.end_node = end_node
        self.weight = weight
        self.edge_type = EdgeType.ATTACK if weight < 0 else EdgeType.SUPPORT


class Layer:
    def __init__(self, nodes: list[Node]):
        self.nodes = nodes
        self.num_nodes = len(nodes)


class Graph:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

        

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
    def _generate_networkx_model(mlp: FFNN, attr_names: list[str]=None) -> tuple[nx.DiGraph, dict]:
        G = nx.DiGraph()

        mlp_shapes = mlp.get_shape()
        mlp_idx_range = []

        # calculate the offset in a list from start to end
        offset = 0
        for shape in mlp_shapes:
            mlp_idx_range.append((offset, offset+shape))
            offset = offset + shape

        if attr_names is None:
            attr_names = [f"X{idx}" for idx in range(mlp_shapes[0])]
        else:
            assert len(attr_names) == mlp_shapes[0]


        # 3 -> 0.99 1.98 2.97
        # 4 -> 0.75 1.50 2.25 3
        # calculate the position of each node
        pos_nodes = {}
        SCALING_FACTOR = 5
        for layer, (start, end) in enumerate(mlp_idx_range):
            num_nodes = end - start + 1
            pos_nodes.update({n : (layer, (1 / num_nodes) * (i + 1) * SCALING_FACTOR) \
                             for i, n in enumerate(range(start, end))})

        # loop through the weights of each layer and add them to the graph
        for l_idx, layer in enumerate(mlp.model.layers):
            weights = layer.get_weights()
            for out_idx, _ in enumerate(weights[0]):
                for in_idx, weight in enumerate(weights[0][out_idx]):
                    cur_layer_node = mlp_idx_range[l_idx][0] + out_idx
                    next_layer_node = mlp_idx_range[l_idx + 1][0] + in_idx
                    G.add_edge(cur_layer_node, next_layer_node, color='r', weight=weight)

        new_labels = {}
        for layer, (start, end) in enumerate(mlp_idx_range):
            if layer == 0:
                new_labels.update({lnode_idx: attr_names[lnode_idx] for lnode_idx in range(len(attr_names))})
            elif layer == len(mlp_idx_range) - 1:
                new_labels.update({start + out_idx: f"O{out_idx}" for out_idx in range(end - start + 1)})
            else:
                new_labels.update({start + lnode_idx: f"C{start + lnode_idx - mlp_idx_range[0][1]}" for \
                    lnode_idx in range(end - start + 1)})

        G = nx.relabel_nodes(G, new_labels)
        new_pos_nodes = {new_labels[node]: pos for (node, pos) in pos_nodes.items()}

        return (G, new_pos_nodes)

    @staticmethod
    def visualise(mlp: FFNN, features: Optional[list[str]] = None) -> None:
        """ Generate interactive visualisation using networkx and bokeh.
        :params
            mlp: FFNN
                the network to be visualised
        """
        (G, pos_nodes) = SimpleVisualizer._generate_networkx_model(mlp, features)

        ATTACK, SUPPORT = "red", "green"
        edge_colors = {}
        edge_weights = {}
        edge_type = {}
        for start_node, end_node, d in G.edges(data=True):
            edge_color = ATTACK if d['weight'] < 0 else SUPPORT
            edge_colors[(start_node, end_node)] = edge_color
            edge_weights[(start_node, end_node)] = d['weight']
            edge_type[(start_node, end_node)] = "Attack" if d['weight'] < 0 else "Support"

        nx.set_edge_attributes(G, edge_colors, "edge_color")
        nx.set_edge_attributes(G, edge_weights, "edge_weight")
        nx.set_edge_attributes(G, edge_type, "edge_type")

        graph = from_networkx(G, pos_nodes)

        node_hover_tool = HoverTool(tooltips=[("index", "@index")], renderers=[graph.node_renderer])
        edge_hover_tool = HoverTool(tooltips=[("edge_weight", "@edge_weight"), ("edge_type", "@edge_type")],
                                    renderers=[graph.edge_renderer], line_policy='interp')


        # plot = Plot(width=WIDTH, height=HEIGHT, x_range=Range1d(-1.1, 1.1),
        #             y_range=Range1d(-1.1, 1.1), sizing_mode='scale_both')
        plot = Plot(sizing_mode='scale_both')

        # add arrows
        # OFFSET_VAL = 0.05
        # for start_node, end_node, d in G.edges(data=True):
        #     arrow = Arrow(end=NormalHead(line_color='black', size=10),
        #           y_start=pos_nodes[start_node][1],
        #           x_start=pos_nodes[start_node][0],
        #           x_end=pos_nodes[end_node][0] - OFFSET_VAL,
        #           y_end=pos_nodes[end_node][1] + OFFSET_VAL if pos_nodes[start_node][1] > pos_nodes[end_node][1] \
        #                                             else pos_nodes[end_node][1] - OFFSET_VAL,
        #           line_width=0)
        #     plot.add_layout(arrow)


        plot.add_tools(node_hover_tool, edge_hover_tool, BoxZoomTool(), ResetTool(), PanTool(), WheelZoomTool())

        graph.edge_renderer.glyph = MultiLine(line_color="edge_color", line_alpha=0.8, line_width="edge_weight")
        graph.node_renderer.glyph = Circle(size=35, fill_color=Spectral4[0])
        plot.renderers.append(graph)

        output_file("networkx_graph.html")
        show(plot, sizing_mode='stretch_both')
