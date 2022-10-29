from abc import abstractmethod
import networkx as nx
from bokeh.io import output_file, show
from bokeh.plotting import from_networkx
from bokeh.models import (BoxZoomTool, HoverTool, Plot, Range1d, ResetTool, PanTool, WheelZoomTool, MultiLine, Circle)
from bokeh.palettes import Spectral4


from ffnn import FFNN

FIG_WIDTH = 1920
FIG_HEIGHT = 1080


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
    def _generate_networkx_model(mlp: FFNN) -> tuple[nx.DiGraph, dict]:
        # TODO: add meaning of labels to the FFNN class
        labels_text = ["petal_length", "petal_width", "sepal_length", "sepal_width"]

        G = nx.DiGraph()

        mlp_shapes = mlp.get_shape()
        mlp_idx_range = []

        # calculate the offset in a list from start to end
        offset = 0
        for shape in mlp_shapes:
            mlp_idx_range.append((offset, offset+shape))
            offset = offset + shape

        # calculate the position of each node
        pos_nodes = {}
        for layer, (start, end) in enumerate(mlp_idx_range):
            #node_dist = FIG_HEIGHT / mlp_shapes[layer]
            pos_nodes.update({n : (layer, i * 2) for i, n in enumerate(range(start, end))})

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
                new_labels.update({lnode_idx: labels_text[lnode_idx] for lnode_idx in range(len(labels_text))})
            elif layer == len(mlp_idx_range) - 1:
                new_labels.update({start + out_idx: f"O{out_idx}" for out_idx in range(end - start + 1)})
            else:
                new_labels.update({start + lnode_idx: f"C{start + lnode_idx - mlp_idx_range[0][1]}" for \
                    lnode_idx in range(end - start + 1)})

        G = nx.relabel_nodes(G, new_labels)
        new_pos_nodes = {new_labels[node]: pos for (node, pos) in pos_nodes.items()}

        return (G, new_pos_nodes)

    @staticmethod
    def visualise(mlp: FFNN) -> None:
        """ Generate interactive visualisation using networkx and bokeh.
        :params
            mlp: FFNN
                the network to be visualised
        """
        (G, pos_nodes) = SimpleVisualizer._generate_networkx_model(mlp)

        plot = Plot(width=1920, height=1080, x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))

        ATTACK, SUPPORT = "red", "green"
        edge_colors = {}
        edge_weights = {}
        for start_node, end_node, d in G.edges(data=True):
            edge_color = ATTACK if d['weight'] < 0 else SUPPORT
            edge_colors[(start_node, end_node)] = edge_color
            edge_weights[(start_node, end_node)] = d['weight']

        nx.set_edge_attributes(G, edge_colors, "edge_color")
        nx.set_edge_attributes(G, edge_weights, "edge_weight")

        graph = from_networkx(G, pos_nodes, scale=0.2, center=(360, 240))

        node_hover_tool = HoverTool(tooltips=[("index", "@index")], renderers=[graph.node_renderer])
        edge_hover_tool = HoverTool(tooltips=[("edge_weight", "@edge_weight")],
                                    renderers=[graph.edge_renderer], line_policy='interp')
        plot.add_tools(node_hover_tool, edge_hover_tool, BoxZoomTool(), ResetTool(), PanTool(), WheelZoomTool())

        graph.edge_renderer.glyph = MultiLine(line_color="edge_color", line_alpha=0.8, line_width="edge_weight")
        graph.node_renderer.glyph = Circle(size=35, fill_color=Spectral4[0])
        plot.renderers.append(graph)

        output_file("networkx_graph.html")
        show(plot)
