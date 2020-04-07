import logging
import os
from statistics import mean, stdev
import numpy as np


class Statistics:
    default_statistics_functions = [len, max, min, mean, stdev,
                                    (np.percentile, [10]), (np.percentile, [20]), (np.percentile, [30]),
                                    (np.percentile, [40]), (np.percentile, [50]), (np.percentile, [60]),
                                    (np.percentile, [70]), (np.percentile, [80]), (np.percentile, [90]),
                                    (np.percentile, [99]), (np.percentile, [99.9]), (np.percentile, [99.99])]

    @staticmethod
    def calculate(data, statistics_functions=None):
        statistics_values = {}
        if statistics_functions is None:
            statistics_functions = Statistics.default_statistics_functions
        for fun in statistics_functions:
            try:
                if isinstance(fun, tuple):
                    statistics_values[fun[0].__name__ + str(fun[1])] = fun[0](data, *fun[1])
                else:
                    statistics_values[fun.__name__] = fun(data)
            except Exception as e:
                print(e)
                if isinstance(fun, tuple):
                    statistics_values[fun[0].__name__ + str(fun[1])] = None
                else:
                    statistics_values[fun.__name__] = None
        return statistics_values

    @staticmethod
    def save(folder, filename, values):
        if not os.path.exists('output/' + folder):
            os.mkdir('output/' + folder)
        with open('output/' + folder + filename, "w+") as file:
            for key in values:
                file.write(str(key) + "," + str(values[key]) + "\n")
                logging.info(str(key) + "," + str(values[key]))
