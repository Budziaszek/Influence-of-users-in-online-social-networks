from Manager import Manager
from Network.GraphConnectionType import GraphConnectionType
from Network.NeighborhoodMode import NeighborhoodMode
from Metrics.MetricsType import MetricsType, GraphIterator
from Utility.Functions import coefficient_of_variation, without_none

from statistics import mean, stdev

manager = Manager(parameters="dbname='salon24' user='sna_user' host='localhost' password='sna_password'", test=False)

static = GraphIterator(GraphIterator.GraphMode.STATIC)
dynamic = GraphIterator(GraphIterator.GraphMode.DYNAMIC)
curr_next = GraphIterator(GraphIterator.GraphMode.DYNAMIC_CURR_NEXT)

c_in = GraphConnectionType.IN
c_out = GraphConnectionType.OUT
c_in_out = [GraphConnectionType.IN, GraphConnectionType.OUT]

connections_in_dynamic = MetricsType(MetricsType.NEIGHBORS_COUNT, c_in, dynamic)
connections_in_static = MetricsType(MetricsType.NEIGHBORS_COUNT, c_in, static)
jaccard_index_neighbors_static = MetricsType(MetricsType.JACCARD_INDEX_NEIGHBORS, c_in_out, static)
jaccard_index_neighbors_curr_next = MetricsType(MetricsType.JACCARD_INDEX_NEIGHBORS, c_in, curr_next)
neighbors_count_difference = MetricsType(MetricsType.NEIGHBORS_COUNT_DIFFERENCE, c_in, curr_next)
new_neighbors = MetricsType(MetricsType.NEW_NEIGHBORS, c_in, curr_next)
composition_1000_6000 = MetricsType(MetricsType.COMPOSITION_NEIGHBORS_PERCENTS, c_in, static, [1001, 6000])
composition_500_1000 = MetricsType(MetricsType.COMPOSITION_NEIGHBORS_PERCENTS, c_in, static, [501, 1000])
composition_100_500 = MetricsType(MetricsType.COMPOSITION_NEIGHBORS_PERCENTS, c_in, static, [101, 500])
composition_0_100 = MetricsType(MetricsType.COMPOSITION_NEIGHBORS_PERCENTS, c_in, static, [0, 100])

modes_to_calculate = [
    NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS
    # Mode.COMMENTS_TO_COMMENTS_FROM_OTHERS,
    # Mode.COMMENTS_TO_POSTS_AND_COMMENTS_FROM_OTHERS,
    # Mode.COMMENTS_TO_POSTS,
    # Mode.COMMENTS_TO_COMMENTS,
    # Mode.COMMENTS_TO_POSTS_AND_COMMENTS
]

values_to_calculate = [
    # connections_in_dynamic,
    # connections_in_static,
    # jaccard_index_neighbors_static,
    # jaccard_index_neighbors_curr_next,
    # neighbors_count_difference,
    # new_neighbors,
    composition_1000_6000,
    composition_500_1000,
    composition_100_500,
    composition_0_100,
]

functions = [
    None,
    mean,
    max,
    stdev,
    # stats.variation
    coefficient_of_variation
]

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
#                           data_condition_function=without_none
#                           )
# HelpFunctions.without_zeros
# x_scale=[0, 1, 2, 6, 11, 21, 31, 51, 61, 71, 81, 91, 101, 151, 201, 251, 501, 6000]
# x_scale=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
# size_scale=[0, 1, 2, 6, 11, 101, 501, 6000]
# x_scale=np.arange(start=0, stop=1.01, step=0.05)
# manager.process_loaded_data(metrics=value, predict=True)

p = ['neighbors_count_static', 'neighbors_count_dynamic', 'jaccard_index_static', 'jaccard_index_dynamic_curr_next']
n_clusters = [3, 6, 10]
for n in n_clusters:
    print('\x1b[3;30;42m' + 'k-means ' + str(n) + '\x1b[0m')
    manager.k_means(6, p)
