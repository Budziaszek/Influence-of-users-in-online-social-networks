import logging
from Manager import Manager
from Metrics.config import *
from Utility.Functions import coefficient_of_variation, without_none, without_nan_none, without_nan
import matplotlib.pyplot as plt

from statistics import mean, stdev

logging.basicConfig(level=logging.INFO)

manager = Manager(parameters="dbname='salon24' user='sna_user' host='localhost' password='sna_password'", test=False)

calculate = False
display = False
correlation = True
ranking = False
cluster = False


if calculate:
    for mode in modes_to_calculate:
        for value in values_to_calculate:
            manager.calculate(mode=mode,
                              save_to_file=False,
                              metrics=value,
                              save_to_database=True,
                              data_condition_function=without_nan
                              )

if display:
    for value in values_to_calculate:
        pass
        manager.histogram(mode=NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS,
                          metrics=value,
                          n_bins=13,
                          # cut=[float("-inf"), float("inf")],
                          cut=[0, float("inf")],
                          half_open=False,
                          integers=False,
                          step=-1,
                          normalize=True
                          )
        # manager.display(mode=NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS,
        #                 metrics=value,
        #                 min=0,
        #                 max=0.030)
        # manager.distribution_linear(mode=NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS,
        #                             metrics=[value],
        #                             cut=(1, 100),
        #                             n_bins=100)
        manager.statistics(mode=NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS,
                           metrics=value, statistics=statistics_functions, normalize=True)
    # manager.points_2d(mode=NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS,
    #                   metrics=values_to_calculate)
    # manager.distribution_linear(mode=NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS,
    #                             metrics=values_to_calculate,
    #                             cut=(500, 1000),
    #                             n_bins=500)
    # manager.histogram(mode=NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS,
    #                   metrics=values_to_calculate,
    #                   n_bins=10,
    #                   cut=[1, -1],
    #                   half_open=False,
    #                   integers=True,
    #                   step=200)

if correlation:
    manager.correlation(NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, values_to_calculate, functions)

if ranking:
    manager.ranking(NeighborhoodMode.COMMENTS_TO_POSTS_AND_COMMENTS_FROM_OTHERS, values_to_calculate)

if cluster:
    n_clusters = [6]

    for n in n_clusters:
        manager.k_means(n, clustering_scenario_1)
