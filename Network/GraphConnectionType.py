from enum import Enum


class GraphConnectionType(Enum):
    IN = "in"
    OUT = "out"
    IN_OUT = "in+out"

    def neighbors_count(self, graph, node):
        if not graph.has_node(node):
            return 0
        if self.value == "out":
            return sum(1 for _ in graph.successors(node))
        if self.value == "in":
            return sum(1 for _ in graph.predecessors(node))
        elif self.value == "in+out":
            return sum(1 for _ in graph.successors(node)) + sum(1 for _ in graph.predecessors(node))
        return 0

    def connections_count(self, graph, node):
        if not graph.has_node(node):
            return 0
        if self.value == "out":
            return sum(graph[u][v]['weight'] for u, v in graph.out_edges(node))
        if self.value == "in":
            return sum(graph[u][v]['weight'] for u, v in graph.in_edges(node))
        elif self.value == "in+out":
            return sum(graph[u][v]['weight'] for u, v in graph.in_edges(node)) \
                   + sum(graph[u][v]['weight'] for u, v in graph.out_edges(node))
        else:
            return 0
