import logging
import os
import sys
from collections import defaultdict
from statistics import median
from numpy import percentile

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
    NEIGHBORHOOD_FRACTION = "neighborhood_fraction"
    NEIGHBORHOOD_QUALITY = "neighborhood_quality"

    # STABILITY
    NEIGHBORS_PER_INTERVAL = "neighbors_per_interval"
    DEGREE_BETWEEN_INTERVALS_DIFFERENCE = "neighbors_count_difference"
    JACCARD_INDEX_INTERVALS = "jaccard_index_intervals"
    NEW_NEIGHBORS = "new_neighbors"
    LOST_NEIGHBORS = "lost_neighbors"
    NEIGHBORS_MAINTENANCE_MEAN = "neighbors_maintenance_mean"
    NEIGHBORS_MAINTENANCE_MAX = "neighbors_maintenance_max"
    NEIGHBORS_MAINTENANCE_MEDIAN = "neighbors_maintenance_median"
    NEIGHBORS_MAINTENANCE_PERCENTILE_90 = "neighbors_maintenance_percentile_90"

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
        NEIGHBORHOOD_FRACTION,
        NEIGHBORHOOD_QUALITY,

        NEIGHBORS_PER_INTERVAL,
        DEGREE_BETWEEN_INTERVALS_DIFFERENCE,
        JACCARD_INDEX_INTERVALS,
        NEW_NEIGHBORS,
        LOST_NEIGHBORS,
        NEIGHBORS_MAINTENANCE_MEAN,
        NEIGHBORS_MAINTENANCE_MAX,
        NEIGHBORS_MAINTENANCE_MEDIAN,
        NEIGHBORS_MAINTENANCE_PERCENTILE_90,

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
        self.last_activity_dates = None
        self.static_degree = None
        if not isinstance(connection_type, list):
            if self.value == self.JACCARD_INDEX_NEIGHBORS:
                self.connection_type = [CONNECTION_IN, CONNECTION_OUT]
            if self.value == self.JACCARD_INDEX_INTERVALS:
                self.connection_type = [connection_type, connection_type]

    def calculate(self, users_ids, first_activity_dates, last_activity_dates, none_before=False, users_selection=None):
        try:
            data = defaultdict(list)
            self.graph_iterator.reset()
            self.users_ids = users_ids
            self.first_activity_dates = first_activity_dates
            # self.last_activity_dates = last_activity_dates
            self.none_before = none_before
            self.users_selection = users_selection
            while not self.graph_iterator.stop:
                graph = self.graph_iterator.next()
                graph_data = self._call_metric_function(self.connection_type, graph, users_selection=users_selection)
                end = graph[1].end_day if isinstance(graph, list) else graph.end_day
                start = graph[1].start_day if isinstance(graph, list) else graph.start_day
                for key in users_ids:
                    if first_activity_dates[key] is None or first_activity_dates[key] <= end:
                        # or last_activity_dates[key] is None \
                        # and last_activity_dates[key] >= start):
                        data[key].append(graph_data[key] if key in graph_data else 0)
                    elif none_before:
                        data[key].append(None)
            return data

        except Exception as e:
            print('Exception saving:', e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

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
        if self.value == self.NEIGHBORHOOD_FRACTION:
            return connection_type.neighborhood_fraction(graph)
        if self.value == self.NEIGHBORHOOD_DENSITY:
            return connection_type.density(graph)
        if self.value == self.NEIGHBORHOOD_QUALITY:
            return connection_type.neighborhood_quality(graph, users_selection)

        # STABILITY
        if self.value == self.NEIGHBORS_PER_INTERVAL:
            return self._neighbors_per_interval(connection_type, graph)
        if self.value == self.DEGREE_BETWEEN_INTERVALS_DIFFERENCE:
            return self._count_difference(connection_type, graph)
        if self.value == self.JACCARD_INDEX_INTERVALS:
            return connection_type[0].jaccard_index_intervals(graph[0], graph[1])
        if self.value == self.NEW_NEIGHBORS:
            return connection_type.neighbors_change(graph[0], graph[1])
        if self.value == self.LOST_NEIGHBORS:
            return connection_type.neighbors_change(graph[0], graph[1], False)
        if self.value == self.NEIGHBORS_MAINTENANCE_MEAN:
            return connection_type.neighbors_maintenance(self.first_activity_dates)
        if self.value == self.NEIGHBORS_MAINTENANCE_MAX:
            return connection_type.neighbors_maintenance(self.first_activity_dates, fun=max)
        if self.value == self.NEIGHBORS_MAINTENANCE_MEDIAN:
            return connection_type.neighbors_maintenance(self.first_activity_dates, fun=median)
        if self.value == self.NEIGHBORS_MAINTENANCE_PERCENTILE_90:
            return connection_type.neighbors_maintenance(self.first_activity_dates, fun=percentile, arg=90)

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
        try:
            if self.static_degree is None:
                self.static_degree = Metrics(Metrics.DEGREE, connection_type, ITERATOR_STATIC) \
                    .calculate(self.users_ids, self.first_activity_dates, self.last_activity_dates,
                               self.none_before, self.users_selection)
            d = connection_type.degree(graph)
            return {key: d[key]/self.static_degree[key][0] if self.static_degree[key][0] != 0 else 0 for key in d}
        except Exception as e:
            print('Exception:', e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    @staticmethod
    def _count_difference(connection_type, graph):
        degree_1 = connection_type.degree(graph[0])
        degree_2 = connection_type.degree(graph[1])
        try:
            r = {key: abs((degree_2[key] if key in degree_2 else 0) - (degree_1[key] if key in degree_1 else 0))
                 for key in GraphIterator.static_graph.nodes}
        except Exception as e:
            print('Exception:', e)
            return None
        return r

