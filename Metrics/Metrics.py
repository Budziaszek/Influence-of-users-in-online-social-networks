import logging
from collections import defaultdict


class Metrics:
    value = None
    complex_description = None
    is_complex = False

    # CENTRALITY
    DEGREE = "degree"  # Number of edges attached (number of neighbors
    WEIGHTED_DEGREE = "weighted_degree"  # Sum of the weights of edges attached
    DEGREE_CENTRALITY = "degree_centrality"  # Degree normalized by dividing by maximum possible degree
    EIGENVECTOR_CENTRALITY = "eigenvector_centrality"
    WEIGHTED_EIGENVECTOR_CENTRALITY = "weighted_eigenvector_centrality"
    # TODO Remember -> unable to calculate katz centrality for full graph!!!
    KATZ_CENTRALITY = "katz_centrality"
    WEIGHTED_KATZ_CENTRALITY = "weighted_katz_centrality"
    CLOSENESS_CENTRALITY = "closeness_centrality"
    BETWEENNESS_CENTRALITY = "betweeness_centrality"

    # TODO Improve metrics
    COMPOSITION_NEIGHBORS_COUNT = "composition_neighbors"
    COMPOSITION_NEIGHBORS_PERCENTS = "composition_neighbors_percents"

    # NEIGHBOTHOOD
    DENSITY = "density"
    RECIPROCITY = "reciprocity"
    JACCARD_INDEX_NEIGHBORS = "jaccard_index"

    # STABILITY
    DIVIDE_NEIGHBORS = "divide_neighbors"
    NEIGHBORS_COUNT_DIFFERENCE = "neighbors_count_difference"
    NEW_NEIGHBORS = "new_neighbors"

    # TODO PageRank
    # TODO local centrality measure
    # TODO neighborhood centrality
    # TODO NL centrality
    # TODO centrality with gravity
    # TODO VoteRank (networkx)

    def get_name(self):
        con = self.connection_type.value if not isinstance(self.connection_type, list) \
            else '_'.join([x.value for x in self.connection_type])
        v = con + "_" + self.value + "_" + '_'.join([str(x) for x in self.graph_iterator.graph_mode])
        return v + "_" + '_'.join([str(x) for x in self.data]) if self.data is not None else v

    def __init__(self, value, connection_type, graph_iterator, data=None):
        self.connection_type = connection_type
        self.graph_iterator = graph_iterator
        self.is_complex = False
        self.value = value
        self.data = data
        if not isinstance(connection_type, list) and self.value is self.JACCARD_INDEX_NEIGHBORS:
            self.connection_type = [connection_type, connection_type]

    def calculate(self, users_ids, first_activity_dates):
        data = defaultdict(list)
        self.graph_iterator.reset()
        while not self.graph_iterator.stop:
            graph = self.graph_iterator.next()
            graph_data = self._call_metric_function(self.connection_type, graph)
            end = graph[1].end_day if isinstance(graph, list) else graph.end_day
            for key in users_ids:
                if first_activity_dates[key] is None or first_activity_dates[key] <= end:
                    data[key].append(graph_data[key] if key in graph_data else 0)
                else:
                    data[key].append(None)
        return data

    def _call_metric_function(self, connection_type, graph):
        # CENTRALITY
        if self.value is self.DEGREE:
            return connection_type.degree(graph)
        if self.value is self.WEIGHTED_DEGREE:
            return connection_type.weighted_degree(graph)
        if self.value is self.DEGREE_CENTRALITY:
            return connection_type.degree_centrality(graph)
        if self.value is self.EIGENVECTOR_CENTRALITY:
            return connection_type.eigenvector_centrality(graph)
        if self.value is self.WEIGHTED_EIGENVECTOR_CENTRALITY:
            return connection_type.eigenvector_centrality(graph, weight=True)
        if self.value is self.KATZ_CENTRALITY:
            return connection_type.katz_centrality(graph)
        if self.value is self.WEIGHTED_KATZ_CENTRALITY:
            return connection_type.katz_centrality(graph, weight=True)
        if self.value is self.CLOSENESS_CENTRALITY:
            return connection_type.closeness_centrality(graph)
        if self.value is self.BETWEENNESS_CENTRALITY:
            return connection_type.betweenness_centrality(graph)

        logging.error('Metrics unimplemented: %s', self.value)

    def _calculate_for_user(self, connection_type, graph, user_id):
        if self.value is self.DEGREE_CENTRALITY:
            return connection_type.old_degree_centrality(graph, user_id)
        if self.value is self.DENSITY:
            return connection_type.density(graph, user_id)
        if self.value is self.RECIPROCITY:
            dictionary = graph.reciprocity([user_id])
            return dictionary[user_id] if user_id in dictionary else 0
        if self.value is self.JACCARD_INDEX_NEIGHBORS:
            return self._jaccard_index(connection_type, graph, user_id)
        if self.value is self.COMPOSITION_NEIGHBORS_COUNT:
            return self._neighborhood_composition(connection_type, graph, user_id, self.data)
        if self.value is self.COMPOSITION_NEIGHBORS_PERCENTS:
            return self._neighborhood_composition(connection_type, graph, user_id, self.data, percent=True)
        if self.value is self.NEIGHBORS_COUNT_DIFFERENCE:
            return self._count_difference(connection_type, graph, user_id)
        if self.value is self.NEW_NEIGHBORS:
            return self._new_neighbors(connection_type, graph, user_id)
        if self.value is self.DIVIDE_NEIGHBORS:
            return connection_type.degree_centrality(graph[0], user_id) \
                   / (connection_type.degree_centrality(graph[1], user_id) + 1)

        logging.error('Metrics unimplemented: %s', self.value)

    @staticmethod
    def _count_difference(connection_type, graph, node):
        neighbors_1 = connection_type.neighbors(graph[0], node)
        neighbors_2 = connection_type.neighbors(graph[1], node)

        return (len(neighbors_2) - len(neighbors_1)) / (len(neighbors_2) + 1)

    @staticmethod
    def _jaccard_index(connection_type, graph, node):
        neighbors_1 = connection_type[0].neighbors(graph[0], node)
        neighbors_2 = connection_type[1].neighbors(graph[1], node)
        if len(neighbors_1) == 0 or len(neighbors_2) == 0:
            return 0
        intersection = list(set(neighbors_1).intersection(set(neighbors_2)))
        return len(intersection) / (len(neighbors_1) + len(neighbors_2) - len(intersection))

    @staticmethod
    def _new_neighbors(connection_type, graph, node):
        neighbors_1 = connection_type.neighbors(graph[0], node)
        neighbors_2 = connection_type.neighbors(graph[1], node)
        if len(neighbors_1) == 0 or len(neighbors_2) == 0:
            return 0
        difference = list(set(neighbors_2).difference(set(neighbors_1)))
        union = list(set(neighbors_1).union(set(neighbors_2)))
        return len(difference) / len(union)

    @staticmethod
    def _neighborhood_composition(connection_type, graph, node, size, percent=False):
        neighbors = connection_type.neighbors(graph, node)
        count = 0
        for neighbor in neighbors:
            neighbors_count = connection_type.degree_centrality(graph, neighbor)
            if neighbors_count >= size[0]:
                if neighbors_count <= size[1]:
                    count += 1
        if percent is True:
            return count / len(neighbors) if len(neighbors) > 0 else 0
        return count

