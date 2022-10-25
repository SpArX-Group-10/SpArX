from abc import abstractmethod
from turtle import color
import networkx as nx
import matplotlib.pyplot as plt

from ffnn import FFNN


class Visualiser:
    """ Base visualisation class """
    @staticmethod
    @abstractmethod
    def visualise(mlp: FFNN) -> None:
        """ visualises the given mlp as a qbaf

        :params
            mlp: FFNN
                the networ to be visualised
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


        # calculate the position of each node
        pos_nodes = {}
        for layer, (start, end) in enumerate(mlp_idx_range):
            pos_nodes.update({n : (layer * 2, i + 0.5) for i, n in enumerate(range(start, end))})


        # loop through the weights of each layer and add them to the graph
        for l_idx, layer in enumerate(mlp.model.layers):
            weights = layer.get_weights()
            for out_idx, _ in enumerate(weights[0]):
                for in_idx, weight in enumerate(weights[0][out_idx]):
                    G.add_edge(mlp_idx_range[l_idx][0] + out_idx, mlp_idx_range[l_idx + 1][0] + in_idx, color='g', weight=weight)


        colors = nx.get_edge_attributes(G,'color').values()
        weights = nx.get_edge_attributes(G,'weight').values()
        nx.draw_networkx(G, pos_nodes, edge_color=list(colors), width=list(weights), with_labels=True, node_color='red')

        # Set margins for the axes so that nodes aren't clipped
        ax = plt.gca()
        ax.margins(0.20)
        plt.show()

