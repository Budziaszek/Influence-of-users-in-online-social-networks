from Manager import Manager
import sys
from App import ManagerApp
from Metrics.config import *
from Utility.Functions import coefficient_of_variation, without_none, without_nan_none, without_nan
import matplotlib.pyplot as plt

from statistics import mean, stdev

manager = Manager(connection_parameters="dbname='salon24' user='sna_user' host='localhost' password='sna_password'", test=False)

calculate = False
display = False
correlation = False
ranking = False
cluster = False

if calculate:
    for neighborhood_mode in modes_to_calculate:
        manager.check_graphs(neighborhood_mode)
        for value in values_to_calculate:
            manager.calculate(save_to_file=False,
                              metrics=value,
                              save_to_database=True,
                              data_condition_function=without_nan,
                              do_users_selection=True if value in [neighborhood_quality_in,
                                                                   neighborhood_quality_out] else False
                              )

if display:
    for value in values_to_calculate:
        pass
        manager.histogram(neighborhood_mode=NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS,
                          metrics=value,
                          n_bins=13,
                          # cut=[float("-inf"), float("inf")],
                          cut=[0, float("inf")],
                          half_open=False,
                          integers=False,
                          step=-1,
                          normalize=True
                          )
        # manager.display_between_range(neighborhood_mode=NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS,
        #                               metrics=value,
        #                               min=0,
        #                               max=0.030)
        # manager.distribution_linear(neighborhood_mode=NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS,
        #                             metrics=[value],
        #                             cut=(1, 100),
        #                             n_bins=100)
        manager.statistics(neighborhood_mode=NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS,
                           metrics=value, statistics=statistics_functions, normalize=True)
    # manager.points_2d(neighborhood_mode=NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS,
    #                   metrics=values_to_calculate)
    # manager.distribution_linear(neighborhood_mode=NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS,
    #                             metrics=values_to_calculate,
    #                             cut=(500, 1000),
    #                             n_bins=500)
    # manager.histogram(neighborhood_mode=NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS,
    #                   metrics=values_to_calculate,
    #                   n_bins=10,
    #                   cut=[1, -1],
    #                   half_open=False,
    #                   integers=True,
    #                   step=200)

if correlation:
    manager.correlation(NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, values_to_calculate, functions)

if ranking:
    manager.table(NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS, values_to_calculate, functions, table_mode="value")

if cluster:
    n_clusters = [6]

    for n in n_clusters:
        manager.k_means(n, clustering_scenario_3,
                        manager.select_users(NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS,
                                             degree_in_static, 1, float('inf')))