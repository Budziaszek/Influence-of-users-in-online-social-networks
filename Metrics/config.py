import statistics
from statistics import stdev, mean

from Metrics.Metrics import Metrics, CONNECTION_IN, ITERATOR_STATIC, CONNECTION_OUT, ITERATOR_DYNAMIC, CONNECTION_IN_OUT
from Network.GraphConnectionType import GraphConnectionType
from Network.GraphIterator import GraphIterator
from Network.NeighborhoodMode import NeighborhoodMode
from Utility.Functions import coefficient_of_variation, max_mode
import numpy as np

degree_in_static = Metrics(Metrics.DEGREE, CONNECTION_IN, ITERATOR_STATIC)
weighted_degree_in_static = Metrics(Metrics.WEIGHTED_DEGREE, CONNECTION_IN, ITERATOR_STATIC)
degree_centrality_in_static = Metrics(Metrics.DEGREE_CENTRALITY, CONNECTION_IN, ITERATOR_STATIC)
eigenvector_centrality_in_static = Metrics(Metrics.EIGENVECTOR_CENTRALITY, CONNECTION_IN, ITERATOR_STATIC)
weighted_eigenvector_centrality_in_static = \
    Metrics(Metrics.WEIGHTED_EIGENVECTOR_CENTRALITY, CONNECTION_IN, ITERATOR_STATIC)
katz_centrality_in_static = Metrics(Metrics.KATZ_CENTRALITY, CONNECTION_IN, ITERATOR_STATIC)
weighted_katz_centrality_in_static = Metrics(Metrics.WEIGHTED_EIGENVECTOR_CENTRALITY, CONNECTION_IN, ITERATOR_STATIC)
closeness_centrality_in_static = Metrics(Metrics.CLOSENESS_CENTRALITY, CONNECTION_IN, ITERATOR_STATIC)
betweenness_centrality_in_static = Metrics(Metrics.BETWEENNESS_CENTRALITY, CONNECTION_IN, ITERATOR_STATIC)
local_centrality_in_static =  Metrics(Metrics.LOCAL_CENTRALITY, CONNECTION_IN, ITERATOR_STATIC)

degree_in_dynamic = Metrics(Metrics.DEGREE, CONNECTION_IN, ITERATOR_DYNAMIC)
weighted_degree_in_dynamic = Metrics(Metrics.WEIGHTED_DEGREE, CONNECTION_IN, ITERATOR_DYNAMIC)
degree_centrality_in_dynamic = Metrics(Metrics.DEGREE_CENTRALITY, CONNECTION_IN, ITERATOR_DYNAMIC)
eigenvector_centrality_in_dynamic = Metrics(Metrics.EIGENVECTOR_CENTRALITY, CONNECTION_IN, ITERATOR_DYNAMIC)
weighted_eigenvector_centrality_in_dynamic = \
    Metrics(Metrics.WEIGHTED_EIGENVECTOR_CENTRALITY, CONNECTION_IN, ITERATOR_DYNAMIC)
katz_centrality_in_dynamic = Metrics(Metrics.KATZ_CENTRALITY, CONNECTION_IN, ITERATOR_DYNAMIC)
weighted_katz_centrality_in_dynamic = Metrics(Metrics.WEIGHTED_EIGENVECTOR_CENTRALITY, CONNECTION_IN, ITERATOR_DYNAMIC)
closeness_centrality_in_dynamic = Metrics(Metrics.CLOSENESS_CENTRALITY, CONNECTION_IN, ITERATOR_DYNAMIC)
betweenness_centrality_in_dynamic = Metrics(Metrics.BETWEENNESS_CENTRALITY, CONNECTION_IN, ITERATOR_DYNAMIC)
local_centrality_in_dynamic =  Metrics(Metrics.LOCAL_CENTRALITY, CONNECTION_IN, ITERATOR_DYNAMIC)

degree_out_static = Metrics(Metrics.DEGREE, CONNECTION_OUT, ITERATOR_STATIC)
weighted_degree_out_static = Metrics(Metrics.WEIGHTED_DEGREE, CONNECTION_OUT, ITERATOR_STATIC)
degree_centrality_out_static = Metrics(Metrics.DEGREE_CENTRALITY, CONNECTION_OUT, ITERATOR_STATIC)
eigenvector_centrality_out_static = Metrics(Metrics.EIGENVECTOR_CENTRALITY, CONNECTION_OUT, ITERATOR_STATIC)
weighted_eigenvector_centrality_out_static = \
    Metrics(Metrics.WEIGHTED_EIGENVECTOR_CENTRALITY, CONNECTION_OUT, ITERATOR_STATIC)
katz_centrality_out_static = Metrics(Metrics.KATZ_CENTRALITY, CONNECTION_OUT, ITERATOR_STATIC)
weighted_katz_centrality_out_static = Metrics(Metrics.WEIGHTED_EIGENVECTOR_CENTRALITY, CONNECTION_OUT, ITERATOR_STATIC)
closeness_centrality_out_static = Metrics(Metrics.CLOSENESS_CENTRALITY, CONNECTION_OUT, ITERATOR_STATIC)
betweenness_centrality_out_static = Metrics(Metrics.BETWEENNESS_CENTRALITY, CONNECTION_OUT, ITERATOR_STATIC)
local_centrality_out_static = Metrics(Metrics.LOCAL_CENTRALITY, CONNECTION_OUT, ITERATOR_STATIC)

degree_out_dynamic = Metrics(Metrics.DEGREE, CONNECTION_OUT, ITERATOR_DYNAMIC)
weighted_degree_out_dynamic = Metrics(Metrics.WEIGHTED_DEGREE, CONNECTION_OUT, ITERATOR_DYNAMIC)
degree_centrality_out_dynamic = Metrics(Metrics.DEGREE_CENTRALITY, CONNECTION_OUT, ITERATOR_DYNAMIC)
eigenvector_centrality_out_dynamic = Metrics(Metrics.EIGENVECTOR_CENTRALITY, CONNECTION_OUT, ITERATOR_DYNAMIC)
weighted_eigenvector_centrality_out_dynamic = \
    Metrics(Metrics.WEIGHTED_EIGENVECTOR_CENTRALITY, CONNECTION_OUT, ITERATOR_DYNAMIC)
katz_centrality_out_dynamic = Metrics(Metrics.KATZ_CENTRALITY, CONNECTION_OUT, ITERATOR_DYNAMIC)
weighted_katz_centrality_out_dynamic = Metrics(Metrics.WEIGHTED_EIGENVECTOR_CENTRALITY, CONNECTION_OUT,
                                               ITERATOR_DYNAMIC)
closeness_centrality_out_dynamic = Metrics(Metrics.CLOSENESS_CENTRALITY, CONNECTION_OUT, ITERATOR_DYNAMIC)
betweenness_centrality_out_dynamic = Metrics(Metrics.BETWEENNESS_CENTRALITY, CONNECTION_OUT, ITERATOR_DYNAMIC)
local_centrality_out_dynamic =  Metrics(Metrics.LOCAL_CENTRALITY, CONNECTION_OUT, ITERATOR_DYNAMIC)

jaccard_index_neighbors_static = Metrics(Metrics.JACCARD_INDEX_NEIGHBORS, CONNECTION_IN_OUT, ITERATOR_STATIC)
reciprocity = Metrics(Metrics.RECIPROCITY, None, ITERATOR_STATIC)
density_neighborhood_in = Metrics(Metrics.NEIGHBORHOOD_DENSITY, CONNECTION_IN, ITERATOR_STATIC)
density_neighborhood_out = Metrics(Metrics.NEIGHBORHOOD_DENSITY, CONNECTION_OUT, ITERATOR_STATIC)
neighborhood_quality_in = Metrics(Metrics.NEIGHBORHOOD_QUALITY, CONNECTION_IN, ITERATOR_STATIC)
neighborhood_quality_out = Metrics(Metrics.NEIGHBORHOOD_QUALITY, CONNECTION_OUT, ITERATOR_STATIC)

# density_in = Metrics(Metrics.DENSITY, CONNECTION_IN, ITERATOR_STATIC)

# composition_1000_6000 = Metrics(Metrics.COMPOSITION_NEIGHBORS_PERCENTS, c_in, static, [1001, 6000])
# composition_500_1000 = Metrics(Metrics.COMPOSITION_NEIGHBORS_PERCENTS, c_in, static, [501, 1000])
# composition_100_500 = Metrics(Metrics.COMPOSITION_NEIGHBORS_PERCENTS, c_in, static, [101, 500])
# composition_0_100 = Metrics(Metrics.COMPOSITION_NEIGHBORS_PERCENTS, c_in, static, [0, 100])
# density_in = Metrics(Metrics.DENSITY, c_in, static)

#
# jaccard_index_neighbors_curr_next = Metrics(Metrics.JACCARD_INDEX_NEIGHBORS, c_in, curr_next)
# neighbors_count_difference = Metrics(Metrics.NEIGHBORS_COUNT_DIFFERENCE, c_in, curr_next)
# new_neighbors = Metrics(Metrics.NEW_NEIGHBORS, c_in, curr_next)
#
# dynamic_divided_by_static = Metrics(Metrics.DIVIDE_NEIGHBORS, c_in, dynamic_static)

modes_to_calculate = [
    NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS
    # NeighborhoodMode.COMMENTS_TO_COMMENTS_FROM_OTHERS,
    # NeighborhoodMode.COMMENTS_TO_POSTS_AND_COMMENTS_FROM_OTHERS,
    # NeighborhoodMode.COMMENTS_TO_POSTS,
    # NeighborhoodMode.COMMENTS_TO_COMMENTS,
    # NeighborhoodMode.COMMENTS_TO_POSTS_AND_COMMENTS
]

values_to_calculate = [
    # degree_in_static,
    # weighted_degree_in_static,
    # # degree_centrality_in_static,
    # eigenvector_centrality_in_static,
    # weighted_eigenvector_centrality_in_static,
    # # # katz_centrality_in_static,
    # # # weighted_katz_centrality_in_static,
    # closeness_centrality_in_static,
    # betweenness_centrality_in_static,
    # local_centrality_in_static,
    # # #
    # degree_in_dynamic,
    # weighted_degree_in_dynamic,
    # # degree_centrality_in_dynamic,
    # eigenvector_centrality_in_dynamic,
    # weighted_eigenvector_centrality_in_dynamic,
    # # # katz_centrality_in_dynamic,
    # # # weighted_katz_centrality_in_dynamic,
    # closeness_centrality_in_dynamic,
    # betweenness_centrality_in_dynamic,
    # local_centrality_in_dynamic,
    #
    # degree_out_static,
    # weighted_degree_out_static,
    # # degree_centrality_out_static,
    # eigenvector_centrality_out_static,
    # weighted_eigenvector_centrality_out_static,
    # # # katz_centrality_out_static,
    # # # weighted_katz_centrality_out_static,
    # closeness_centrality_out_static,
    # betweenness_centrality_out_static,
    # local_centrality_out_static,
    #
    # degree_out_dynamic,
    # weighted_degree_out_dynamic,
    # # degree_centrality_out_dynamic,
    # eigenvector_centrality_out_dynamic,
    # weighted_eigenvector_centrality_out_dynamic,
    # # katz_centrality_out_dynamic,
    # # weighted_katz_centrality_out_dynamic,
    # closeness_centrality_out_dynamic,
    # betweenness_centrality_out_dynamic,
    # local_centrality_out_dynamic,
    #
    # jaccard_index_neighbors_static,
    # reciprocity,
    # density_neighborhood_in,
    # density_neighborhood_out,
    neighborhood_quality_in,
    neighborhood_quality_out,

]

functions = [
    # None,
    sum,
    mean,
    max,
    min,
    max_mode,
    statistics.median,

    # np.std,
    # stdev,
    # stats.variation
    # coefficient_of_variation,
]

clustering_parameters = [
    (NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, weighted_degree_in_static, 1),
    (NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, weighted_degree_out_static, 1),
    # (reciprocity, 1),
    # (jaccard_index, 1)
]


clustering_scenario_1 = [
    (NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, degree_in_static, 1),
    (NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, weighted_degree_in_static, 1),
    (NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, degree_out_static, 1),
    (NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, weighted_degree_out_static, 1),
    (NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, reciprocity, 1),
    (NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, jaccard_index_neighbors_static, 1)
]

clustering_scenario_2 = [
    (NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, closeness_centrality_in_static, 1),
    (NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, betweenness_centrality_in_static, 1),
    (NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, local_centrality_in_static, 1),
    (NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, weighted_eigenvector_centrality_in_static, 1),

    (NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, closeness_centrality_in_dynamic, 1, statistics.median),
    (NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, betweenness_centrality_in_dynamic, 1, statistics.median),
    (NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, local_centrality_in_dynamic, 1, statistics.median),
    (NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, weighted_eigenvector_centrality_in_dynamic, 1, statistics.median),
]

clustering_scenario_3 = [
    (NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, degree_in_static, 1),
    (NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, weighted_eigenvector_centrality_in_static, 1),
    (NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, closeness_centrality_in_static, 1),

    (NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, density_neighborhood_in, 1),
    (NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, neighborhood_quality_in, 1),
    (NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, reciprocity, 1),
]

clustering_scenario_4 = [

]

statistics_functions = [len, max, mean, stdev, #(np.percentile, [20]),
                        (np.percentile, [10]), (np.percentile, [30]), (np.percentile, [50]),
                        (np.percentile, [70]), (np.percentile, [75]), (np.percentile, [80]),
                        (np.percentile, [85]), (np.percentile, [90]), (np.percentile, [95]),
                        (np.percentile, [99]), (np.percentile, [99.9]), (np.percentile, [99.99])]