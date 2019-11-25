from statistics import mean, stdev
import numpy as np

class Statistics:

    def __init__(self):
        self.statistics_values = {}
        self.default_statistics_functions = [(len, None), (max, None), (min, None), (mean, None), (stdev, None),
                                             (np.percentile, [10]), (np.percentile, [20]), (np.percentile, [30]),
                                             (np.percentile, [40]), (np.percentile, [50]), (np.percentile, [60]),
                                             (np.percentile, [70]), (np.percentile, [80]), (np.percentile, [90]),
                                             (np.percentile, [99]), (np.percentile, [99.9]), (np.percentile, [99.99])]

    def calculate(self, data, statistics_functions=None):
        if statistics_functions is None:
            statistics_functions = self.default_statistics_functions
        for fun in statistics_functions:
            try:
                if fun[1] is None:
                    self.statistics_values[fun[0].__name__] = fun[0](data)
                else:
                    self.statistics_values[fun[0].__name__ + str(fun[1])] = fun[0](data, *fun[1])
            except Exception:
                self.statistics_values[fun[0].__name__] = None

    def save(self, folder, filename):
        with open(folder + filename, "w+") as file:
            for key in self.statistics_values:
                file.write(str(key) + "," + str(self.statistics_values[key]) + "\n")
