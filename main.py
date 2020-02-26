from Manager import Manager
from Network.GraphConnectionType import GraphConnectionType
from Network.NeighborhoodMode import NeighborhoodMode
from Metrics.MetricsType import MetricsType, GraphIterator
from statistics import mean, stdev

manager = Manager(parameters="dbname='salon24' user='sna_user' host='localhost' password='sna_password'", test=False)

modes_to_calculate = [
    NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS
    # Mode.COMMENTS_TO_COMMENTS_FROM_OTHERS,
    # Mode.COMMENTS_TO_POSTS_AND_COMMENTS_FROM_OTHERS,
    # Mode.COMMENTS_TO_POSTS,
    # Mode.COMMENTS_TO_COMMENTS,
    # Mode.COMMENTS_TO_POSTS_AND_COMMENTS
]

values_to_calculate = [
    MetricsType(MetricsType.NEIGHBORS_COUNT, GraphConnectionType.IN, GraphIterator(GraphIterator.GraphMode.DYNAMIC)),
    MetricsType(MetricsType.NEIGHBORS_COUNT, GraphConnectionType.IN, GraphIterator(GraphIterator.GraphMode.STATIC))
    # MetricsType(MetricsType.JACCARD_INDEX_NEIGHBORS, [GraphConnectionType.IN, GraphConnectionType.OUT])
    # MetricsType(MetricsType.JACCARD_INDEX_NEIGHBORS, GraphConnectionType.IN,
    #             GraphIterator(GraphIterator.GraphMode.DYNAMIC_CURR_NEXT)),
    # MetricsType(MetricsType.NEIGHBORS_COUNT_DIFFERENCE, GraphConnectionType.IN,
    #             GraphIterator(GraphIterator.GraphMode.DYNAMIC_CURR_NEXT))
    # MetricsType(MetricsType.NEW_NEIGHBORS, GraphConnectionType.IN,
    #             GraphIterator(GraphIterator.GraphMode.DYNAMIC_CURR_NEXT))
    # MetricsType(MetricsType.COMPOSITION_NEIGHBORS_PERCENTS, GraphConnectionType.IN,
    #             GraphIterator(GraphIterator.GraphMode.STATIC), [1001, 6000]),
    # MetricsType(MetricsType.COMPOSITION_NEIGHBORS_PERCENTS, GraphConnectionType.IN,
    #             GraphIterator(GraphIterator.GraphMode.STATIC), [501, 1000]),
    # MetricsType(MetricsType.COMPOSITION_NEIGHBORS_PERCENTS, GraphConnectionType.IN,
    #             GraphIterator(GraphIterator.GraphMode.STATIC), [101, 500]),
    # MetricsType(MetricsType.COMPOSITION_NEIGHBORS_PERCENTS, GraphConnectionType.IN,
    #             GraphIterator(GraphIterator.GraphMode.STATIC), [0, 100])
]


def coefficient_of_variation(data):
    if len(data) > 1 and mean(data) > 0:
        return stdev(data) / mean(data)
    else:
        return 0


functions = [
    None,
    mean,
    max,
    stdev,
    # stats.variation
    coefficient_of_variation
]

for mode in modes_to_calculate:
    for value in values_to_calculate:
        # manager.calculate(mode=mode, save_to_file=False, calculate_histogram=True, predict=False, calculated_value=value,
        #                   x_scale=np.arange(start=0, stop=1.01, step=0.05),
        #                   size_scale=[0, 1, 2, 6, 11, 101, 501, 6000], data_functions=functions,
        #                   data_condition_function=None)
        # HelpFunctions.without_zeros
        # x_scale=[0, 1, 2, 6, 11, 21, 31, 51, 61, 71, 81, 91, 101, 151, 201, 251, 501, 6000]
        # x_scale=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        # size_scale=[0, 1, 2, 6, 11, 101, 501, 6000]
        # x_scale=np.arange(start=0, stop=1.01, step=0.05)
        manager.process_loaded_data(metrics=value, predict=True)
