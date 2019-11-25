from Manager import Manager
from Network.GraphConnectionType import GraphConnectionType
from Mode import Mode
from ComplexMeasure import ComplexMeasure, MeasureOperation, BasicMeasure, MeasureGraphIterator
from CustomMetricsTypeDesctiption import CustomMetricsTypeDescription
from HelpFunctions import HelpFunctions
import numpy as np
from statistics import mean, stdev
from scipy import stats

manager = Manager(parameters="dbname='salon24' user='sna_user' host='localhost' password='sna_password'")

modes_to_calculate = [
    Mode.COMMENTS_TO_POSTS_FROM_OTHERS
    # Mode.COMMENTS_TO_COMMENTS_FROM_OTHERS,
    # Mode.COMMENTS_TO_POSTS_AND_COMMENTS_FROM_OTHERS,
    # Mode.COMMENTS_TO_POSTS,
    # Mode.COMMENTS_TO_COMMENTS,
    # Mode.COMMENTS_TO_POSTS_AND_COMMENTS
]
neighbors_in = BasicMeasure(BasicMeasure.NEIGHBORS, GraphConnectionType.IN)

iterator_0 = MeasureGraphIterator(neighbors_in, MeasureGraphIterator.GraphMode.DYNAMIC, 0, -1)
iterator_1 = MeasureGraphIterator(neighbors_in, MeasureGraphIterator.GraphMode.DYNAMIC, 1)

neighbors_intersection = MeasureOperation(iterator_0.get_name(),
                                          MeasureOperation.SetOperation.INTERSECTION,
                                          iterator_1.get_name())
neighbors_union = MeasureOperation(iterator_0.get_name(),
                                   MeasureOperation.SetOperation.UNION,
                                   iterator_1.get_name())
neighbors_intersection_length = MeasureOperation(neighbors_intersection.get_name(),
                                                 MeasureOperation.SetReduction.LENGTH)
neighbors_union_length = MeasureOperation(neighbors_union.get_name(), MeasureOperation.SetReduction.LENGTH)
jaccard = MeasureOperation(neighbors_intersection_length.get_name(), MeasureOperation.ValueOperator.DIVISION,
                           neighbors_union_length.get_name())

JACCARD_INDEX_DYNAMIC = ComplexMeasure("jaccard_index_dynamic",
                                       [neighbors_intersection, neighbors_union, neighbors_intersection_length,
                                        neighbors_union_length, jaccard],
                                       [iterator_0, iterator_1])

neighbors_length = MeasureOperation(neighbors_in.get_name(), MeasureOperation.SetReduction.LENGTH)
A = ComplexMeasure("A_new",
                   [MeasureOperation("neighbors_in_dynamic0", MeasureOperation.SetReduction.LENGTH),
                    MeasureOperation("neighbors_in_static", MeasureOperation.SetReduction.LENGTH),
                    MeasureOperation("neighbors_in_dynamic0_length", MeasureOperation.ValueOperator.DIVISION,
                                     "neighbors_in_static_length")],
                   [MeasureGraphIterator(neighbors_in, MeasureGraphIterator.GraphMode.DYNAMIC, 0),
                    MeasureGraphIterator(neighbors_in, MeasureGraphIterator.GraphMode.STATIC)])
NEIGHBORHOOD_IN_SIZE = ComplexMeasure("neighborhood_in",
                                      [MeasureOperation(neighbors_in.get_name() + "_static",
                                                        MeasureOperation.SetReduction.LENGTH)],
                                      [MeasureGraphIterator(neighbors_in, MeasureGraphIterator.GraphMode.STATIC)])
values_to_calculate = [
    # NEIGHBORHOOD_IN_SIZE
    A,
    JACCARD_INDEX_DYNAMIC
]


def coefficient_of_variation(data):
    if len(data) > 1:
        return stdev(data) / mean(data)
    else:
        return 0


functions = [
    # None,
    # mean,
    # max,
    # stdev,
    # stats.variation,
    coefficient_of_variation
]

for mode in modes_to_calculate:
    #  TODO split statistics_values into parts
    manager.generate_graph_data(mode=mode)
    for value in values_to_calculate:
        manager.calculate(calculate_full_data=False, calculate_histogram=True, calculated_value=value,
                          x_scale=np.arange(start=0, stop=1.01, step=0.05),
                          size_scale=[0, 1, 2, 6, 11, 101, 501, 6000], data_functions=functions,
                          data_condition_function=HelpFunctions.without_zeros)
        # HelpFunctions.without_zeros
        # x_scale=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        # size_scale=[0, 1, 2, 5, 10, 100, 500, 6000])
