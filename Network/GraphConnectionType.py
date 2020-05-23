from collections import defaultdict
from enum import Enum


class GraphConnectionType:
    value = None

    IN = "in"
    OUT = "out"

    CONNECTION_TYPES = [
        IN,
        OUT,
    ]

    def __init__(self, value):
        self.value = value

    def degree(self, graph):
        if self.value == self.IN:
            return graph.in_degree()
        elif self.value == self.OUT:
            return graph.out_degree()
        else:
            return graph.degree()

    def degree_centrality(self, graph):
        if self.value == self.IN:
            return graph.in_degree_centrality()
        elif self.value == self.OUT:
            return graph.out_degree_centrality()
        else:
            return graph.degree_centrality()

    def weighted_degree(self, graph):
        if self.value == self.IN:
            return graph.in_degree(True)
        elif self.value == self.OUT:
            return graph.out_degree(True)
        else:
            return graph.degree(True)

    def neighbors(self, graph):
        d = defaultdict(list)
        for node in graph.nodes:
            if not graph.has_node(node):
                d[node] = []
            elif self.value == self.OUT:
                d[node] = graph.successors(node)
            elif self.value == self.IN:
                d[node] = graph.predecessors(node)
        return d

    def density(self, graph):
        return graph.density(graph.successors if self.value == self.OUT else graph.predecessors)

    def eigenvector_centrality(self, graph, weight=False):
        if self.value == self.IN:
            return graph.eigenvector_centrality(weight=weight)
        elif self.value == self.OUT:
            return graph.eigenvector_centrality(weight=weight, reverse=True)

    def katz_centrality(self, graph, weight=False):
        if self.value == self.IN:
            return graph.katz_centrality(weight=weight)
        elif self.value == self.OUT:
            return graph.katz_centrality(weight=weight, reverse=True)

    def closeness_centrality(self, graph):
        if self.value == self.IN:
            return graph.closeness_centrality()
        elif self.value == self.OUT:
            return graph.closeness_centrality(reverse=True)

    def local_centrality(self, graph):
        if self.value == self.IN:
            return graph.local_centrality()
        elif self.value == self.OUT:
            return graph.local_centrality(in_neighborhood=False)

