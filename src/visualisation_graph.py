from enum import Enum

class EdgeType(Enum):
    """Specifies the edge type after clustering."""
    ATTACK = 'red'
    SUPPORT = 'green'

class Node:
    """The building block of a neural network."""
    def __init__(self, idx: int, x: float, y: float, feature_name: str, supports:dict[str, float]={}):
        self.idx = idx
        self.x = x
        self.y = y
        self.feature_name = feature_name
        self.supports = supports
        
    def rename(self, name: str):
        self.feature_name=name
        
    def transfer_attack_support(self, supports: dict[str, float], weight: float):
        for n,v in supports.items():
            self.supports.update({n: self.supports.get(n, 0.0)+v*weight})
            
    def get_supporting_nodes(self):
        res = set()
        for n,v in self.supports.items():
            if v>0:
                res.add(n)
        return res
    
    def get_attacking_nodes(self):
        res = set()
        for n,v in self.supports.items():
            if v<0:
                res.add(n)
        return res
            

class Edge:
    """Connects one node to the other between two adjacent layers."""
    def __init__(self, start_node: Node, end_node: Node, weight: float):
        self.start_node = start_node
        self.end_node = end_node
        self.weight = weight
        self.edge_type = EdgeType.ATTACK if weight < 0 else EdgeType.SUPPORT


# class Layer:
#     """Contains a series of nodes in the neural network."""
#     def __init__(self, nodes: list[Node]):
#         self.nodes = nodes
#         self.num_nodes = len(nodes)


class Graph:
    """Neural network in matrix form."""
    def __init__(self, nodes: dict[str, Node], edges: list[Edge]):
        self.nodes = nodes
        self.edges = edges
        
    def get_node(self, name: str)->Node:
        return self.nodes.get(name)
