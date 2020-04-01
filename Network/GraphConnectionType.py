from enum import Enum


class GraphConnectionType(Enum):
    IN = "in"
    OUT = "out"
    IN_OUT = "in+out"

    def old_degree_centrality(self, graph, node):
        return len(self.neighbors(graph, node))

    def degree(self, graph):
        if self.value == self.IN.value:
            return graph.in_degree()
        elif self.value == self.OUT.value:
            return graph.out_degree()
        else:
            return graph.degree()

    def degree_centrality(self, graph):
        if self.value == self.IN.value:
            return graph.in_degree_centrality()
        elif self.value == self.OUT.value:
            return graph.out_degree_centrality()
        else:
            return graph.degree_centrality()

    def weighted_degree(self, graph):
        if self.value == self.IN.value:
            return graph.in_degree(True)
        elif self.value == self.OUT.value:
            return graph.out_degree(True)
        else:
            return graph.degree(True)

    def neighbors(self, graph, node, value=None):
        if value is None:
            value = self.value
        if not graph.has_node(node):
            return []
        if value == self.OUT.value:
            return [n for n in graph.successors(node)]
        elif value == self.IN.value:
            return [n for n in graph.predecessors(node)]
        # elif value == self.IN_OUT.value:
        #     return list(set([n for n in graph.successors(node)]).union([n for n in graph.predecessors(node)]))
        return []

    def connections(self, graph, node):
        if not graph.has_node(node):
            return []
        if self.value == self.OUT.value:
            return [graph[u][v][self._G.nodes] for u, v in graph.out_edges(node)]
        elif self.value == self.IN.value:
            return [graph[u][v]['weight'] for u, v in graph.in_edges(node)]
        else:
            return []

    def density(self, graph, node):
        neighbors = self.neighbors(graph, node)
        # Remove considered node
        if node in neighbors:
            neighbors.remove(node)
        # Ignore neighborhood if user have no neighbors or neighborhood to small
        if len(neighbors) == 0 or len(neighbors) == 1:
            return 0
        #  Initialize values - numbers of users in neighborhood connected to user (does NOT include user itself)
        values = []
        for n in neighbors:
            n_neighbors = self.neighbors(graph, n)
            values.append(len(list(set(n_neighbors).intersection(neighbors))))

        size = (len(neighbors) - 1)*len(neighbors)
        value = sum(values)/size
        return value

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
        neighbors_in = self.neighbors(graph, node, self.IN.value)
        neighbors_out = self.neighbors(graph, node, self.OUT.value)
        union = list(set(neighbors_in).union(set(neighbors_out)))
        if len(union) == 0:
            return 0
        elif self.value == self.IN.value:
            return len(neighbors_in)/len(union)
        elif self.value == self.OUT.value:
            return len(neighbors_out)/len(union)



