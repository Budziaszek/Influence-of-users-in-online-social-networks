import networkx as nx
import matplotlib.pyplot as plt


class NetworkEngine:
    def __init__(self, is_multi):
        if is_multi is True:
            self.G = nx.MultiDiGraph()
        else:
            self.G = nx.DiGraph()

    def add_edges(self, edges):
        if self.G.is_multigraph():
            self.G.add_edges_from(edges)
        else:
            for edge in edges:
                if self.G.has_edge(*edge):
                    self.G.add_edge(*edge, weight=self.G.get_edge_data(*edge)['weight']+1)
                else:
                    self.G.add_edge(*edge, weight=1)
        # nx.draw(self.G)
        # plt.show()
