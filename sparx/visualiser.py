from abc import abstractmethod
from typing import Optional
import networkx as nx
from bokeh.io import output_file, show
from bokeh.plotting import from_networkx
from bokeh.models import BoxZoomTool, HoverTool, Plot, ResetTool, PanTool, WheelZoomTool, MultiLine, Circle
from bokeh.palettes import Spectral4

from .ffnn import FFNN
from .visualisation_graph import Edge, Graph, Node, Layer


def generate_nodes(
    mlp_idx_range: list[tuple[int]], edges: list[tuple[int, int, float]], attr_names: list[str] = None
) -> tuple[dict, dict[str, Node], list[Edge], list[Layer]]:
    """Generate positions and nodes for visualisation"""
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
            pos_nodes.update({n: (x_pos, y_pos)})
            name = attr_names.get(n)
            new_node = None
            if layer == 0:
                new_node = Node(n, x=x_pos, y=y_pos, label=name, layer=0, incoming={name: 1.0})
            else:
                new_node = Node(n, x=x_pos, y=y_pos, label=name, layer=layer)
            nodes.update({name: new_node})
            layer_nodes.append(new_node)
            edges = [(new_node, end, w) if start == n else (start, end, w) for start, end, w in edges]
            edges = [(start, new_node, w) if end == n else (start, end, w) for start, end, w in edges]
        layers.append(Layer(layer_nodes))

    vis_graph_edges = [Edge(start, end, w) for start, end, w in edges]

    return pos_nodes, nodes, vis_graph_edges, layers


def generate_weights(mlp: FFNN, G: nx.DiGraph, mlp_idx_range: list[tuple[int]]) -> list[tuple[int, int, float]]:
    """Generate weights for visualisation"""
    edges = []
    for l_idx, layer in enumerate(mlp.model.layers):
        weights = layer.get_weights()
        for out_idx, _ in enumerate(weights[0]):
            for in_idx, weight in enumerate(weights[0][out_idx]):
                cur_layer_node = mlp_idx_range[l_idx][0] + out_idx
                next_layer_node = mlp_idx_range[l_idx + 1][0] + in_idx
                edges.append((cur_layer_node, next_layer_node, weight))
                G.add_edge(cur_layer_node, next_layer_node, color="r", weight=weight)

    return edges


def relabel_nodes(mlp_idx_range: list[tuple[int]], attr_names: list[str]) -> dict:
    """Relabel nodes given the feature names, layer type and index"""
    new_labels = {}
    for layer, (start, end) in enumerate(mlp_idx_range):
        if layer == 0:
            new_labels.update({lnode_idx: attr_names[lnode_idx] for lnode_idx in range(len(attr_names))})
        elif layer == len(mlp_idx_range) - 1:
            new_labels.update({start + out_idx: f"O{out_idx}" for out_idx in range(end - start + 1)})
        else:
            new_labels.update(
                {
                    start + lnode_idx: f"C{start + lnode_idx - mlp_idx_range[0][1]}"
                    for lnode_idx in range(end - start + 1)
                }
            )
    return new_labels


def get_attack_support_by_node(
    graph: Graph, propagate: bool = False
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
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
    """Base visualisation class"""

    @staticmethod
    @abstractmethod
    def visualise(mlp: FFNN) -> None:
        """visualises the given mlp as a qbaf

        :params
            mlp: FFNN
                the network to be visualised
        """
        raise NotImplementedError


class JSONVisualizer(Visualiser):
    """Generates a basic representation of the clustered model using networkx"""

    @staticmethod
    def _generate_networkx_model(
        mlp: FFNN, attr_names: list[str] = None
    ) -> tuple[nx.DiGraph, Graph, dict, list[Layer]]:
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
        """Generate interactive visualisation using networkx and bokeh.
        :params
            mlp: FFNN
                the network to be visualised
        """

        # TODO: add layer as return value if needed
        (_, custom_graph, _, _) = JSONVisualizer._generate_networkx_model(mlp, features)
        get_attack_support_by_node(custom_graph)

        # FOR COMM
        return custom_graph.toJSON()


class BokehVisualizer(Visualiser):
    """Generates a basic representation of the clustered model using networkx"""

    @staticmethod
    def _generate_networkx_model(
        mlp: FFNN, attr_names: list[str] = None
    ) -> tuple[nx.DiGraph, Graph, dict, list[Layer]]:
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
        """Generate interactisve visualisation using networkx and bokeh.
        :params
            mlp: FFNN
                the network to be visualised
        """
        # TODO: add layer as return value if needed
        (G, graph, pos_nodes, _) = BokehVisualizer._generate_networkx_model(mlp, features)

        ATTACK, SUPPORT = "red", "green"
        edge_colors = {}
        edge_weights = {}
        edge_type = {}
        for start_node_idx, end_node_idx, d in G.edges(data=True):
            edge_colors[(start_node_idx, end_node_idx)] = ATTACK if d["weight"] < 0 else SUPPORT
            edge_weights[(start_node_idx, end_node_idx)] = d["weight"]
            edge_type[(start_node_idx, end_node_idx)] = "Attack" if d["weight"] < 0 else "Support"

        supports, attacks = get_attack_support_by_node(graph)

        supports = {label: ", ".join(supporting_nodes) for (label, supporting_nodes) in supports.items()}
        attacks = {label: ", ".join(attacking_nodes) for (label, attacking_nodes) in attacks.items()}

        nx.set_node_attributes(G, supports, "supports")
        nx.set_node_attributes(G, attacks, "attacks")
        nx.set_edge_attributes(G, edge_colors, "edge_color")
        nx.set_edge_attributes(G, edge_weights, "edge_weight")
        nx.set_edge_attributes(G, edge_type, "edge_type")

        graph = from_networkx(G, pos_nodes)

        node_hover_tool = HoverTool(
            tooltips=[("index", "@index"), ("supports", "@supports"), ("attacks", "@attacks")],
            renderers=[graph.node_renderer],
        )
        edge_hover_tool = HoverTool(
            tooltips=[("edge_weight", "@edge_weight"), ("edge_type", "@edge_type")],
            renderers=[graph.edge_renderer],
            line_policy="interp",
        )

        # plot = Plot(width=WIDTH, height=HEIGHT, x_range=Range1d(-1.1, 1.1),
        #             y_range=Range1d(-1.1, 1.1), sizing_mode='scale_both')
        plot = Plot(sizing_mode="scale_both")

        plot.add_tools(node_hover_tool, edge_hover_tool, BoxZoomTool(), ResetTool(), PanTool(), WheelZoomTool())

        graph.edge_renderer.glyph = MultiLine(line_color="edge_color", line_alpha=0.8, line_width="edge_weight")
        graph.node_renderer.glyph = Circle(size=35, fill_color=Spectral4[0])
        plot.renderers.append(graph)

        output_file("networkx_graph.html")
        show(plot, sizing_mode="stretch_both")
