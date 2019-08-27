from enum import Enum


class GraphConnectionType(Enum):
    IN = "in"
    OUT = "out"
    IN_OUT = "in+out"

    def neighbors_count(self, graph, node):
        len(self.neighbors(graph, node))
        # if not graph.has_node(node):
        #     return 0
        # if self.value == "out":
        #     return sum(1 for _ in graph.successors(node))
        # elif self.value == "in":
        #     return sum(1 for _ in graph.predecessors(node))
        # elif self.value == "in+out":
        #     return sum(1 for _ in graph.successors(node)) + sum(1 for _ in graph.predecessors(node))
        # return 0

    def neighbors(self, graph, node):
        if not graph.has_node(node):
            return []
        if self.value == "out":
            return [n for n in graph.successors(node)]
        elif self.value == "in":
            return [n for n in graph.predecessors(node)]
        elif self.value == "in+out":
            return [n for n in graph.successors(node)] + [n for n in graph.predecessors(node)]
        return []

    def connections_count(self, graph, node):
        if not graph.has_node(node):
            return 0
        if self.value == "out":
            return sum(graph[u][v]['weight'] for u, v in graph.out_edges(node))
        elif self.value == "in":
            return sum(graph[u][v]['weight'] for u, v in graph.in_edges(node))
        elif self.value == "in+out":
            return sum(graph[u][v]['weight'] for u, v in graph.in_edges(node)) \
                   + sum(graph[u][v]['weight'] for u, v in graph.out_edges(node))
        else:
            return 0

    def connections_strength(self, graph, node, neighborhoods_by_size):
        neighbors = self.neighbors(graph, node)
        if len(neighbors) == 0:
            return 0
        neighbors.append(node)  # Include also considered node
        #  Initialize values - numbers of users in neighborhood connected to user (does NOT include user itself)
        values = []
        for n in neighbors:
            n_neighbors = self.neighbors(graph, n)
            values.append(len(list(set(n_neighbors).intersection(neighbors))) / len(neighbors) * 100)

        size = len(neighbors)
        value = sum(values)/size
        # if size in neighborhoods_by_size.keys():
        neighborhoods_by_size[size].append(value)
        # else:
        #     nneighborhoods_by_size[neighbors] = [value]
        return value
