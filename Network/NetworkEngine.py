import networkx as nx
import matplotlib.pyplot as plt


class NetworkEngine:
    G = nx.MultiDiGraph()

    def add_edges(self, edges):
        self.G.add_edges_from(edges)
        # nx.draw(self.G)
        # plt.show()
