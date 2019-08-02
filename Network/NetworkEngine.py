import networkx as nx
import matplotlib.pyplot as plt


class NetworkEngine:
    def __init__(self, is_multi):
        if is_multi is True:
            self.G = nx.MultiDiGraph()
        else:
            self.G = nx.DiGraph()

    def add_edges(self, edges):
        # Add multi edges
        if self.G.is_multigraph():
            self.G.add_edges_from(edges)
        # Add single edge with number of connections as weight
        else:
            for edge in edges:
                data = self.G.get_edge_data(*edge)
                self.G.add_edge(*edge, weight=int(0 if data is None else data['weight'])+1)
        # nx.draw(self.G)
        # plt.show()
