from collections import defaultdict
from statistics import mean

from code.network.GraphIterator import GraphIterator


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

    def neighborhood_fraction(self, graph):
        return graph.neighborhood_fraction(graph.successors if self.value == self.OUT else graph.predecessors)

    def neighborhood_quality(self, graph, users_selection):
        return graph.neighborhood_quality(graph.successors if self.value == self.OUT else graph.predecessors,
                                          users_selection)

    def neighbors_change(self, graph_1, graph_2, check_for_new=True):
        return dict(self._neighbors_change_iter(graph_1, graph_2, check_for_new))

    def _neighbors_change_iter(self, graph_1, graph_2, check_for_new):
        """ Return an iterator of (node, neighborhood_quality).
        """
        f_1 = graph_1.successors if self.value == self.OUT else graph_1.predecessors
        f_2 = graph_2.successors if self.value == self.OUT else graph_2.predecessors
        for node in GraphIterator.static_graph.nodes:
            neigh_1 = set(f_1(node))
            neigh_2 = set(f_2(node))
            diff = neigh_2 - neigh_1 if check_for_new else neigh_1 - neigh_2
            union = neigh_1 | neigh_2

            if len(union) == 0:
                yield (node, 0)
            else:
                change = len(diff) / len(union)
                yield (node, change)

    def jaccard_index_intervals(self,  graph_1, graph_2):
        return dict(self._jaccard_index_iter(graph_1, graph_2))

    def _jaccard_index_iter(self, graph_1, graph_2):
        """ Return an iterator of (node, jaccard_index).
        """
        f_1 = graph_1.successors if self.value == self.OUT else graph_1.predecessors
        f_2 = graph_2.successors if self.value == self.OUT else graph_2.predecessors
        for node in GraphIterator.static_graph.nodes:
            neigh_1 = set(f_1(node))
            neigh_2 = set(f_2(node))
            intersection = neigh_1 & neigh_2

            if len(neigh_1) == 0 or len(neigh_2) == 0:
                yield (node, 0)
            else:
                jaccard_index = len(intersection)/(len(neigh_1) + len(neigh_2) - len(intersection))
                yield (node, jaccard_index)

    def neighbors_maintenance(self, first_activity_dates, fun=mean, arg=None):
        return dict(self._neighbors_maintenance_iter(first_activity_dates, fun, arg))

    def _neighbors_maintenance_iter(self, first_activity_dates, fun, arg):
        for node in GraphIterator.static_graph.nodes:
            maintenance = {}
            activity = {}
            neighbors = list(GraphIterator.static_graph.successors(node)) if self.value == self.OUT \
                else list(GraphIterator.static_graph.predecessors(node))
            if len(neighbors) > 0:
                for neighbor in neighbors:
                    maintenance[neighbor] = 0
                    activity[neighbor] = 0
                for graph in GraphIterator.dynamic_graphs:
                    if first_activity_dates[node] < graph.end_day:
                        continue
                    n = list(graph.successors(node)) if self.value == self.OUT else list(graph.predecessors(node))
                    for neighbor in neighbors:
                        if first_activity_dates[neighbor] > graph.end_day:
                            continue
                        activity[neighbor] += 1
                        if neighbor in n:
                            maintenance[neighbor] += 1
            if len(maintenance.values()) > 0:
                if arg is None:
                    maintenance_fun = fun([maintenance[key] / activity[key] if activity[key] > 0 else 0
                                           for key in maintenance.keys()])
                else:
                    maintenance_fun = fun([maintenance[key] / activity[key] if activity[key] > 0 else 0
                                           for key in maintenance.keys()], arg)
                yield (node, maintenance_fun)
            else:
                yield (node, 0)
