from abc import abstractmethod
from turtle import color
import networkx as nx
import matplotlib.pyplot as plt

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
    """ Simple visualisation using networkx"""
    @staticmethod
    def visualise(mlp: FFNN) -> None:
        G = nx.DiGraph()

        mlp_shapes = mlp.get_shape()
        mlp_idx_range = []

        # calculate the offset in a list from start to end
        offset = 0
        for shape in mlp_shapes:
            mlp_idx_range.append((offset, offset+shape))
            offset = offset+shape

        #layer_size = FIG_WIDTH / len(mlp_idx_range)

        # calculate the position of each node
        pos_nodes = {}
        for layer, (start, end) in enumerate(mlp_idx_range):
            #node_dist = FIG_HEIGHT / mlp_shapes[layer]
            pos_nodes.update({n : (layer, i * 2) for i, n in enumerate(range(start, end))})


        # loop through the weights of each layer and add them to the graph
        for l_idx, layer in enumerate(mlp.model.layers):
            weights = layer.get_weights()
            print(f"layer {l_idx} weights are {weights[0]}")
            for out_idx, _ in enumerate(weights[0]):
                for in_idx, weight in enumerate(weights[0][out_idx]):
                    cur_layer_node = mlp_idx_range[l_idx][0] + out_idx
                    next_layer_node = mlp_idx_range[l_idx + 1][0] + in_idx
                    G.add_edge(cur_layer_node, next_layer_node, color='r', weight=weight)
                    #G.edges[cur_layer_node, next_layer_node]['color'] = "green"
                    #G[cur_layer_node][next_layer_node]["weight"] = weight
  
        edges = G.edges()
        colors = ['g' if G[u][v]['weight'] >= 0 else 'r' for (u, v) in edges]
        weights = [abs(G[u][v]['weight']) for (u, v) in edges]
        labels = dict([((u, v), round(G[u][v]['weight'], 2)) for (u, v) in edges])
        nodes = nx.draw_networkx_nodes(G, pos_nodes)
        nx.draw(G, pos_nodes, edge_color=colors, font_color='black', width=weights, node_color='w', with_labels=True)
        nodes.set_edgecolor('b')
        nx.draw_networkx_edge_labels(G, pos_nodes, edge_labels=labels, font_color='black', font_size=8, bbox=dict(alpha=0), label_pos = 0.75)
        # Set margins for the axes so that nodes aren't clipped
        ax = plt.gca()
        ax.margins(0.20)
        plt.show()

