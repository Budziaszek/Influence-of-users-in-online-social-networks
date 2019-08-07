import networkx as nx
import matplotlib.pyplot as plt


class SocialNetworkGraph:
    def __init__(self, is_multi):
        if is_multi is True:
            self._G = nx.MultiDiGraph()
        else:
            self._G = nx.DiGraph()
        self.nodes = self._G.nodes

    def __getitem__(self, i):
        return self._G[i]

    def __len__(self):
        return len(self._G)

    def add_edges(self, edges):
        # Add multi edges
        if self._G.is_multigraph():
            self._G.add_edges_from(edges)
        # Add single edge with number of connections as weight
        else:
            for edge in edges:
                data = self._G.get_edge_data(*edge)
                self._G.add_edge(*edge, weight=int(0 if data is None else data['weight'])+1)
        # nx.draw(self.G)
        # plt.show()

    def successors(self, node):
        return self._G.successors(node)

    def predecessors(self, node):
        return self._G.predecessors(node)

    def has_node(self, node):
        return self._G.has_node(node)

    def out_edges(self, node):
        return self._G.out_edges(node)

    def in_edges(self, node):
        return self._G.in_edges(node)
