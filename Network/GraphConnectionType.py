from enum import Enum


class GraphConnectionType:
    value = None

    IN = "in"
    OUT = "out"
    IN_OUT = "in+out"

    CONNECTION_TYPES = [
        IN,
        OUT,
        # IN_OUT
    ]

    def __init__(self, value):
        self.value = value

    def old_degree_centrality(self, graph, node):
        return len(self.neighbors(graph, node))

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

    def neighbors(self, graph, node, value=None):
        if value is None:
            value = self.value
        if not graph.has_node(node):
            return []
        if value == self.OUT:
            return [n for n in graph.successors(node)]
        elif value == self.IN:
            return [n for n in graph.predecessors(node)]
        # elif value == self.IN_OUT.value:
        #     return list(set([n for n in graph.successors(node)]).union([n for n in graph.predecessors(node)]))
        return []

    def connections(self, graph, node):
        if not graph.has_node(node):
            return []
        if self.value == self.OUT:
            return [graph[u][v][graph.nodes] for u, v in graph.out_edges(node)]
        elif self.value == self.IN:
            return [graph[u][v]['weight'] for u, v in graph.in_edges(node)]
        else:
            return []

    def density(self, graph):
        return graph.density(graph.successors if self.value == self.OUT else graph.predecessors)

    def intersection(self, graph_1, graph_2, node):
        neighbors_graph_1 = self.neighbors(graph_1, node)
        neighbors_graph_2 = self.neighbors(graph_2, node)
        if (len(neighbors_graph_1) + len(neighbors_graph_2)) == 0:
            return 0
        return list(set(neighbors_graph_1).intersection(set(neighbors_graph_2)))

    def union(self, graph_1, graph_2, node):
        neighbors_graph_1 = self.neighbors(graph_1, node)
        neighbors_graph_2 = self.neighbors(graph_2, node)
        if (len(neighbors_graph_1) + len(neighbors_graph_2)) == 0:
            return 0
        return list(set(neighbors_graph_1).union(set(neighbors_graph_2)))

    def part_of_neighborhood(self, graph, node):
        neighbors_in = self.neighbors(graph, node, self.IN)
        neighbors_out = self.neighbors(graph, node, self.OUT)
        union = list(set(neighbors_in).union(set(neighbors_out)))
        if len(union) == 0:
            return 0
        elif self.value == self.IN:
            return len(neighbors_in) / len(union)
        elif self.value == self.OUT:
            return len(neighbors_out) / len(union)

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

    def betweenness_centrality(self, graph):
        if self.value == self.IN:
            return graph.betweenness_centrality()
        elif self.value == self.OUT:
            return graph.betweenness_centrality(reverse=True)

    def local_centrality(self, graph):
        if self.value == self.IN:
            return graph.local_centrality()
        elif self.value == self.OUT:
            return graph.local_centrality(in_neighborhood=False)

