from Manager import Manager
from Network.GraphConnectionType import GraphConnectionType
from Mode import Mode
from MetricsType import MetricsType
from CustomMetricsTypeDesctiption import CustomMetricsTypeDescription
from HelpFunctions import HelpFunctions
import numpy as np
from statistics import mean, stdev

manager = Manager(parameters="dbname='salon24' user='sna_user' host='localhost' password='sna_password'")

modes_to_calculate = [
    Mode.COMMENTS_TO_POSTS_FROM_OTHERS
    # Mode.COMMENTS_TO_COMMENTS_FROM_OTHERS,
    # Mode.COMMENTS_TO_POSTS_AND_COMMENTS_FROM_OTHERS,
    # Mode.COMMENTS_TO_POSTS,
    # Mode.COMMENTS_TO_COMMENTS,
    # Mode.COMMENTS_TO_POSTS_AND_COMMENTS
]

description = CustomMetricsTypeDescription(MetricsType(MetricsType.NEIGHBORS_COUNT), "/",
                                           MetricsType(MetricsType.NEIGHBORS_COUNT), "0")
connections_dynamic_divided_by_static = MetricsType(MetricsType.CUSTOM_COMPLEX, description)

values_to_calculate = [
    # MetricsType(MetricsType.CONNECTIONS_COUNT),
    # MetricsType(MetricsType.NEIGHBORS_COUNT),
    # MetricsType(MetricsType.DENSITY),
    # MetricsType(MetricsType.RECIPROCITY),
    # MetricsType(MetricsType.PART_IN_UNION),
    # MetricsType(MetricsType.JACCARD_INDEX),
    connections_dynamic_divided_by_static
]
connections_to_calculate = [
    GraphConnectionType.IN,
    # GraphConnectionType.OUT
]

functions = [
    None,
    mean,
    max,
    stdev
]

for mode in modes_to_calculate:
    #  TODO split data into parts
    manager.generate_graph_data(mode=mode, graph_type="sd", is_multi=False)
    for value in values_to_calculate:
        for connection in connections_to_calculate:
            for fun in functions:
                manager.calculate(calculate_full_data=False, calculate_histogram=True, calculated_value=value,
                                  connection_type=connection, x_scale=np.arange(start=0, stop=1.01, step=0.05),
                                  size_scale=[0, 1, 2, 6, 11, 101, 501, 6000], data_function=fun,
                                  data_condition_function=HelpFunctions.without_zeros, mode="d")

            # x_scale=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            # size_scale=[0, 1, 2, 5, 10, 100, 500, 6000])
