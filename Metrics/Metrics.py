import logging
from collections import defaultdict

from Network.GraphConnectionType import GraphConnectionType
from Network.GraphIterator import GraphIterator

CONNECTION_IN = GraphConnectionType(GraphConnectionType.IN)
CONNECTION_OUT = GraphConnectionType(GraphConnectionType.OUT)
CONNECTION_IN_OUT = [GraphConnectionType(GraphConnectionType.IN), GraphConnectionType(GraphConnectionType.OUT)]

ITERATOR_STATIC = GraphIterator(GraphIterator.ITERATOR.STATIC)
ITERATOR_DYNAMIC = GraphIterator(GraphIterator.ITERATOR.DYNAMIC)
ITERATOR_CURRENT_NEXT = GraphIterator(GraphIterator.ITERATOR.DYNAMIC_CURR_NEXT)
ITERATOR_DYNAMIC_AND_STATIC = GraphIterator([GraphIterator.ITERATOR.DYNAMIC, GraphIterator.ITERATOR.STATIC])


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
    BETWEENNESS_CENTRALITY = "betweenness_centrality"
    LOCAL_CENTRALITY = "local_centrality"

    # NEIGHBORHOOD
    NEIGHBORHOOD_DENSITY = "neighborhood_density"
    RECIPROCITY = "reciprocity"
    JACCARD_INDEX_NEIGHBORS = "jaccard_index"
    NEIGHBORHOOD_QUALITY = "neighborhood_quality"

    # STABILITY
    NEIGHBORS_PER_INTERVAL = "neighbors_per_interval"
    DEGREE_BETWEEN_INTERVALS_DIFFERENCE = "neighbors_count_difference"
    JACCARD_INDEX_INTERVALS = "jaccard_index_intervals"
    NEW_NEIGHBORS = "new_neighbors"
    LOST_NEIGHBORS = "lost_neighbors"

    # ACTIVITY
    POSTS_ADDED = 'posts_added'
    RESPONSES_ADDED = 'responses_added'
    RESPONSES_PER_POST_ADDED = 'responses_per_post_added'

    METRICS_LIST = [
        DEGREE,
        WEIGHTED_DEGREE,
        DEGREE_CENTRALITY,
        EIGENVECTOR_CENTRALITY,
        WEIGHTED_EIGENVECTOR_CENTRALITY,
        KATZ_CENTRALITY,
        WEIGHTED_KATZ_CENTRALITY,
        CLOSENESS_CENTRALITY,
        BETWEENNESS_CENTRALITY,
        LOCAL_CENTRALITY,

        NEIGHBORHOOD_DENSITY,
        RECIPROCITY,
        JACCARD_INDEX_NEIGHBORS,
        NEIGHBORHOOD_QUALITY,

        NEIGHBORS_PER_INTERVAL,
        DEGREE_BETWEEN_INTERVALS_DIFFERENCE,
        JACCARD_INDEX_INTERVALS,
        NEW_NEIGHBORS,
        LOST_NEIGHBORS,

        POSTS_ADDED,
        RESPONSES_ADDED,
        RESPONSES_PER_POST_ADDED,
    ]

    # TODO PageRank
    # TODO neighborhood centrality
    # TODO centrality with gravity
    # TODO VoteRank (networkx)

    def get_name(self):
        con = ""
        if self.connection_type is not None:
            con = self.connection_type.value if not isinstance(self.connection_type, list) \
                else '_'.join([x.value for x in self.connection_type])
        return con + "_" + self.value + "_" + '_'.join([str(x) for x in self.graph_iterator.graph_mode])

    def __init__(self, value, connection_type, graph_iterator):
        self.connection_type = connection_type
        self.graph_iterator = graph_iterator
        self.is_complex = False
        self.value = value
        self.users_ids = None
        self.none_before = False
        self.users_selection = None
        self.first_activity_dates = None
        self.static_degree = None
        if not isinstance(connection_type, list):
            if self.value == self.JACCARD_INDEX_NEIGHBORS:
                self.connection_type = [CONNECTION_IN, CONNECTION_OUT]
            if self.value == self.JACCARD_INDEX_INTERVALS:
                self.connection_type = [connection_type, connection_type]

    def calculate(self, users_ids, first_activity_dates, none_before=False, users_selection=None):
        data = defaultdict(list)
        self.graph_iterator.reset()
        self.users_ids = users_ids
        self.first_activity_dates = first_activity_dates
        self.none_before = none_before
        self.users_selection = users_selection
        while not self.graph_iterator.stop:
            print("Next")
            graph = self.graph_iterator.next()
            graph_data = self._call_metric_function(self.connection_type, graph, users_selection=users_selection)
            end = graph[1].end_day if isinstance(graph, list) else graph.end_day
            for key in users_ids:
                if first_activity_dates[key] is None or first_activity_dates[key] <= end:
                    data[key].append(graph_data[key] if key in graph_data else 0)
                elif none_before:
                    data[key].append(None)
        return data

    def _call_metric_function(self, connection_type, graph, users_selection=None):
        # CENTRALITY
        if self.value == self.DEGREE:
            return connection_type.degree(graph)
        if self.value == self.WEIGHTED_DEGREE:
            return connection_type.weighted_degree(graph)
        if self.value == self.DEGREE_CENTRALITY:
            return connection_type.degree_centrality(graph)
        if self.value == self.EIGENVECTOR_CENTRALITY:
            return connection_type.eigenvector_centrality(graph)
        if self.value == self.WEIGHTED_EIGENVECTOR_CENTRALITY:
            return connection_type.eigenvector_centrality(graph, weight=True)
        if self.value == self.KATZ_CENTRALITY:
            return connection_type.katz_centrality(graph)
        if self.value == self.WEIGHTED_KATZ_CENTRALITY:
            return connection_type.katz_centrality(graph, weight=True)
        if self.value == self.CLOSENESS_CENTRALITY:
            return connection_type.closeness_centrality(graph)
        if self.value == self.BETWEENNESS_CENTRALITY:
            return graph.betweenness_centrality()
        if self.value == self.LOCAL_CENTRALITY:
            return connection_type.local_centrality(graph)

        # NEIGHBORHOOD
        if self.value == self.RECIPROCITY:
            return graph.reciprocity()
        if self.value == self.JACCARD_INDEX_NEIGHBORS:
            return graph.jaccard_index_neighborhoods()
        if self.value == self.NEIGHBORHOOD_DENSITY:
            return connection_type.density(graph)
        if self.value == self.NEIGHBORHOOD_QUALITY:
            return self._neighborhood_composition(connection_type, graph, users_selection=users_selection, percent=True)

        # STABILITY
        if self.value == self.NEIGHBORS_PER_INTERVAL:
            return self._neighbors_per_interval(connection_type, graph)
        if self.value == self.DEGREE_BETWEEN_INTERVALS_DIFFERENCE:
            return self._count_difference(connection_type, graph)
        if self.value == self.JACCARD_INDEX_INTERVALS:
            return self._jaccard_index(connection_type, graph)
        if self.value == self.NEW_NEIGHBORS:
            return self._neighbors_change(connection_type, graph)
        if self.value == self.LOST_NEIGHBORS:
            return self._neighbors_change(connection_type, graph, False)

        # ACTIVITY
        if self.value == self.POSTS_ADDED:
            return graph.get_nodes_attribute('posts')
        if self.value == self.RESPONSES_ADDED:
            return graph.get_nodes_attribute('responses')
        if self.value == self.RESPONSES_PER_POST_ADDED:
            p = graph.get_nodes_attribute('posts')
            r = graph.get_nodes_attribute('responses')
            return {k: r[k]/p[k] if k in r and p[k] > 0 else 0 for k in p}


        logging.error('Metrics unimplemented: %s', self.value)

    def _neighbors_per_interval(self, connection_type, graph):
        if self.static_degree is None:
            self.static_degree = Metrics(Metrics.DEGREE, connection_type, ITERATOR_STATIC) \
                .calculate(self.users_ids, self.first_activity_dates, self.none_before, self.users_selection)
        d = connection_type.degree(graph)
        return {key: d[key]/self.static_degree[key][0] if self.static_degree[key][0] != 0 else 0 for key in d}

    @staticmethod
    def _count_difference(connection_type, graph):
        degree_1 = connection_type.degree(graph[0])
        degree_2 = connection_type.degree(graph[1])
        try:
            keys = set(degree_2.keys()).intersection(degree_1.keys())
            r = {key: (degree_2[key] - degree_1[key])/(degree_2[key] + 1) for key in keys}
        except Exception as e:
            print('Exception:', e)
            return None
        return r

    @staticmethod
    def _jaccard_index(connection_type, graph):
        try:
            jaccard_index = {}
            if not isinstance(graph, list):
                nodes = graph.nodes
                graph = [graph, graph]
            else:
                nodes = set(graph[0].nodes).union(set(graph[1].nodes))
            if not isinstance(connection_type, list):
                connection_type = [connection_type, connection_type]
            for node in nodes:
                neighbors_1 = connection_type[0].neighbors(graph[0], node)
                if len(neighbors_1) == 0:
                    jaccard_index[node] = 0
                    continue
                neighbors_2 = connection_type[1].neighbors(graph[1], node)
                if len(neighbors_2) == 0:
                    jaccard_index[node] = 0
                    continue
                intersection = list(set(neighbors_1).intersection(set(neighbors_2)))
                jaccard_index[node] = len(intersection) / (len(neighbors_1) + len(neighbors_2) - len(intersection))
            return jaccard_index
        except Exception as e:
            print(e)
            print(graph)

    @staticmethod
    def _neighbors_change(connection_type, graph, new=True):
        neighbors_1 = connection_type.neighbors(graph[0])
        neighbors_2 = connection_type.neighbors(graph[1])
        d = defaultdict()
        for key in neighbors_1:
            if len(neighbors_1[key]) == 0 or len(neighbors_2[key]) == 0:
                d[key] = 0
            else:
                if new:
                    difference = list(set(neighbors_2[key]).difference(set(neighbors_1[key])))
                else:
                    difference = list(set(neighbors_1[key]).difference(set(neighbors_2[key])))
                union = list(set(neighbors_1[key]).union(set(neighbors_2[key])))
                d[key] = len(difference) / len(union)
        return d

    @staticmethod
    def _neighborhood_composition(connection_type, graph, users_selection, percent=False):
        composition = {}
        for node in graph.nodes:
            neighbors = list(connection_type.neighbors(graph, node))
            count = len(set(users_selection).intersection(set(neighbors)))
            if percent is True:
                composition[node] = count / len(users_selection) if len(users_selection) > 0 else 0
            else:
                composition[node] = count
        return composition
