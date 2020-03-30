from Manager import Manager
from Network.GraphConnectionType import GraphConnectionType
from Network.GraphIterator import GraphIterator
from Network.NeighborhoodMode import NeighborhoodMode
from Metrics.Metrics import Metrics
from Utility.Functions import coefficient_of_variation, without_none, without_nan_none

from statistics import mean, stdev

manager = Manager(parameters="dbname='salon24' user='sna_user' host='localhost' password='sna_password'", test=False)

static = GraphIterator([GraphIterator.GraphMode.STATIC])
dynamic = GraphIterator([GraphIterator.GraphMode.DYNAMIC])
curr_next = GraphIterator([GraphIterator.GraphMode.DYNAMIC_CURR_NEXT])
dynamic_static = GraphIterator([GraphIterator.GraphMode.DYNAMIC, GraphIterator.GraphMode.STATIC])

c_in = GraphConnectionType.IN
c_out = GraphConnectionType.OUT
c_in_out = [GraphConnectionType.IN, GraphConnectionType.OUT]

connections_in_dynamic = Metrics(Metrics.WEIGHTED_DEGREE_CENTRALITY, c_in, dynamic)
neighbors_count_in_dynamic = Metrics(Metrics.DEGREE_CENTRALITY, c_in, dynamic)
connections_out_dynamic = Metrics(Metrics.WEIGHTED_DEGREE_CENTRALITY, c_out, dynamic)
neighbors_count_out_dynamic = Metrics(Metrics.DEGREE_CENTRALITY, c_out, dynamic)

connections_in_static = Metrics(Metrics.WEIGHTED_DEGREE_CENTRALITY, c_in, static)
neighbors_count_in_static = Metrics(Metrics.DEGREE_CENTRALITY, c_in, static)
connections_out_static = Metrics(Metrics.WEIGHTED_DEGREE_CENTRALITY, c_out, static)
neighbors_count_out_static = Metrics(Metrics.DEGREE_CENTRALITY, c_out, static)

jaccard_index_neighbors_static = Metrics(Metrics.JACCARD_INDEX_NEIGHBORS, c_in_out, static)
composition_1000_6000 = Metrics(Metrics.COMPOSITION_NEIGHBORS_PERCENTS, c_in, static, [1001, 6000])
composition_500_1000 = Metrics(Metrics.COMPOSITION_NEIGHBORS_PERCENTS, c_in, static, [501, 1000])
composition_100_500 = Metrics(Metrics.COMPOSITION_NEIGHBORS_PERCENTS, c_in, static, [101, 500])
composition_0_100 = Metrics(Metrics.COMPOSITION_NEIGHBORS_PERCENTS, c_in, static, [0, 100])
density_in = Metrics(Metrics.DENSITY, c_in, static)
reciprocity_in = Metrics(Metrics.RECIPROCITY, c_in, static)

jaccard_index_neighbors_curr_next = Metrics(Metrics.JACCARD_INDEX_NEIGHBORS, c_in, curr_next)
neighbors_count_difference = Metrics(Metrics.NEIGHBORS_COUNT_DIFFERENCE, c_in, curr_next)
new_neighbors = Metrics(Metrics.NEW_NEIGHBORS, c_in, curr_next)

dynamic_divided_by_static = Metrics(Metrics.DIVIDE_NEIGHBORS, c_in, dynamic_static)

modes_to_calculate = [
    NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS
    # Mode.COMMENTS_TO_COMMENTS_FROM_OTHERS,
    # Mode.COMMENTS_TO_POSTS_AND_COMMENTS_FROM_OTHERS,
    # Mode.COMMENTS_TO_POSTS,
    # Mode.COMMENTS_TO_COMMENTS,
    # Mode.COMMENTS_TO_POSTS_AND_COMMENTS
]

values_to_calculate = [
    connections_in_static,
    connections_out_static,
    connections_in_dynamic,
    connections_out_dynamic,
    neighbors_count_in_dynamic,
    neighbors_count_out_dynamic,
    neighbors_count_in_static,
    neighbors_count_out_static,
    jaccard_index_neighbors_static,
    jaccard_index_neighbors_curr_next,
    neighbors_count_difference,
    new_neighbors,
    # composition_1000_6000,
    # composition_500_1000,
    # composition_100_500,
    # composition_0_100,
    density_in,
    reciprocity_in,
    dynamic_divided_by_static
]

functions = [
    None,
    mean,
    max,
    stdev,
    # stats.variation
    coefficient_of_variation
]
#
# for mode in modes_to_calculate:
#     for value in values_to_calculate:
#         manager.calculate(mode=mode,
#                           save_to_file=False,
#                           calculate_histogram=False,
#                           predict=False,
#                           metrics=value,
#                           save_to_database=True,
#                           # x_scale=np.arange(start=0, stop=1.01, step=0.05),
#                           # size_scale=[0, 1, 2, 6, 11, 101, 501, 6000], data_functions=functions,
#                           data_condition_function=without_nan_none
#                           )

# HelpFunctions.without_zeros
# x_scale=[0, 1, 2, 6, 11, 21, 31, 51, 61, 71, 81, 91, 101, 151, 201, 251, 501, 6000]
# x_scale=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
# size_scale=[0, 1, 2, 6, 11, 101, 501, 6000]
# x_scale=np.arange(start=0, stop=1.01, step=0.05)
# manager.process_loaded_data(metrics=value, predict=True)

p = [
    ('po_in_neighbors_count_static', 1),
    ('po_in_neighbors_count_dynamic', 1),

    ('po_in_connections_count_static', 1),
    ('po_in_connections_count_dynamic', 1),

    ('po_out_neighbors_count_static', 1),
    ('po_out_neighbors_count_dynamic', 1),

    ('po_out_connections_count_static', 1),
    ('po_out_connections_count_dynamic', 1),

    # ('composition_neighbors_percents_static_1001_6000', 1),
    # ('composition_neighbors_percents_static_501_1000', 1),
    # ('composition_neighbors_percents_static_101_500', 1),
    # ('composition_neighbors_percents_static_0_100', 1),
    # ('jaccard_index_static', 1),
    #
    # ('jaccard_index_dynamic_curr_next', 1),
    # ('neighbors_count_difference_dynamic_curr_next', 1),
    # ('new_neighbors_dynamic_curr_next', 1),
    # ('divide_neighbors_dynamic_static', 1),
    #
    # ('density_static', 1),
    # ('reciprocity_static', 1),
]

n_clusters = [10]

for n in n_clusters:
    manager.k_means(n, p)
