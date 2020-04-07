import networkx as nx
import matplotlib.pyplot as plt
from networkx import DiGraph


class SocialNetworkGraph:
    start_day = None
    end_day = None

    def __init__(self):
        self._G = nx.DiGraph()
        self.nodes = self._G.nodes

    def __getitem__(self, i):
        return self._G[i]

    def __len__(self):
        return len(self._G)

    def add_edges(self, edges):
        for edge in edges:
            data = self._G.get_edge_data(*edge)
            self._G.add_edge(*edge, weight=int(0 if data is None else data['weight']) + 1)

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

    def number_of_nodes(self):
        return self._G.number_of_nodes()

    def has_edge(self, u, v):
        return self._G.has_edge(u, v)

    def reciprocity(self, nodes, neighborhood_limit=None):
        if neighborhood_limit is None:
            return nx.algorithms.reciprocity(self._G, nodes)

    def degree(self, weight=False):
        if weight:
            return {node: sum([self._G[u][v]['weight'] for u, v in self._G.edges(node)]) for node in self.nodes}
        return {n: d for n, d in list(self._G.degree(self._G.nodes))}

    def in_degree(self, weight=False):
        if weight:
            return {node: sum([self._G[u][v]['weight'] for u, v in self._G.in_edges(node)]) for node in self.nodes}
        return {n: d for n, d in list(self._G.in_degree(self._G.nodes))}

    def out_degree(self, weight=False):
        if weight:
            return {node: sum([self._G[u][v]['weight'] for u, v in self._G.out_edges(node)]) for node in self.nodes}
        return {n: d for n, d in list(self._G.out_degree(self._G.nodes))}

    def in_degree_centrality(self):
        return nx.in_degree_centrality(self._G)

    def out_degree_centrality(self):
        return nx.out_degree_centrality(self._G)

    def degree_centrality(self):
        return nx.degree_centrality(self._G)

    def eigenvector_centrality(self, weight=False, reverse=False):
        if not weight:
            return nx.eigenvector_centrality(self._G if not reverse else self._G.reverse(), max_iter=1000)
        return nx.eigenvector_centrality(self._G if not reverse else self._G.reverse(), weight='weight', max_iter=1000)

    def katz_centrality(self, weight=False, reverse=False):
        if not weight:
            return nx.katz_centrality(self._G if not reverse else self._G.reverse(), max_iter=5000)
        return nx.katz_centrality(self._G if not reverse else self._G.reverse(), weight='weight', max_iter=5000)

    def closeness_centrality(self, reverse=False):
        return nx.closeness_centrality(self._G if not reverse else self._G.reverse())

    def betweenness_centrality(self, reverse=False):
        # TODO check if there is a difference
        return nx.betweenness_centrality(self._G if not reverse else self._G.reverse())

