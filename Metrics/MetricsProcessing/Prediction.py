import datetime
from statistics import mean

import pandas as pd
from collections import defaultdict
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima_model import ARIMA


class Prediction:

    def __init__(self, data, user_name):
        # List of measures in dictionary (key - station name, value - list of measures)
        self.data = data
        # DataFrame with 'Orginal' column and others columns (if imputed)
        # x label (dates) are used as index (df.index)
        self.data_frame = None
        self._create_series()
        self.user_name = user_name

    def _create_series(self):
        index = [i for i in range(len(self.data))]
        df = pd.DataFrame(list(zip(index, self.data)), columns=['Index', 'Original'])
        df = df.set_index('Index')
        self.data_frame = df

    @staticmethod
    def plot(title, data, labels):
        fig, ax = plt.subplots()
        colors = ['black', 'red', 'green', 'blue', 'purple', 'yellow', 'pink']
        for i, d in enumerate(data):
            ax.plot(d[0], d[1], '-', color=colors[i])  # Solid line _data
        plt.legend(labels, loc='upper right')
        plt.title(title)
        plt.show()

    @staticmethod
    def mean_error(expected, predicted):
        return math.fabs((expected - predicted))

    @staticmethod
    def MAPE_error(expected, predicted):
        return math.fabs((expected - predicted) / expected)

    def predict(self, start, end, test_length, fun, error_fun, parameters_version=None):
        error_data = []

        data_original = self.data_frame[start:end + test_length]
        data_train = self.data_frame[start:end]
        data_test = self.data_frame[end:end + test_length]

        # print(len(data_original), len(data_train), len(data_test))

        series_train = pd.Series(data_train.Original, data_train.index)
        series_original = pd.Series(data_original.Original, data_original.index)

        fitted_values, predicted = fun(series_train, test_length, parameters_version)

        for e, p in zip(data_test.Original, predicted):
            error_data.append(error_fun(e, p))

            # if fitted_values is not None:
            #     result_plot = (series_train.index, fitted_values, data_test.index, predicted)
            # else:

        result_plot = (data_test.index, predicted)
        plot_data = []
        title_data = []

        # plot_data.append((series_original.index, series_original))
        # title_data.append("Original")

        plot_data.append(result_plot)
        title_data.append(fun.__name__)

        return fun.__name__, result_plot, series_original, error_data
        # self.plot("Prediction " + fun.__name__,
        #           [(series_original.index, series_original),
        #            result_plot],
        #           ("Original _data", "Fitted values", "Predicted values"))

    @staticmethod
    def exponential_smoothing(series_train, test_length, parameters_version=None):
        if parameters_version is 1:
            fit = ExponentialSmoothing(series_train, trend="add").fit(use_brute=True)
        elif parameters_version is 2:
            fit = ExponentialSmoothing(series_train, trend="mul").fit(use_brute=True)
        # elif parameters_version is 3:
        #     fit = ExponentialSmoothing(series_train, seasonal='add', trend='add').fit(use_brute=True)
        # elif parameters_version is 4:
        #     fit = ExponentialSmoothing(series_train, seasonal='mul', trend='add').fit(use_boxcox=True)
        else:
            fit = ExponentialSmoothing(series_train).fit()

        fitted_values = fit.fittedvalues
        predicted = fit.forecast(test_length)

        return fitted_values.values, predicted.values

    @staticmethod
    def ARIMA(series_train, test_length, parameters_version=None):
        fit = ARIMA(series_train, order=(5, 1, 0)).fit(disp=0)

        fitted_values = None
        predicted = fit.forecast(test_length)

        return fitted_values, predicted[0]

    @staticmethod
    def mean(series_train, test_length, parameters_version=None):
        predicted = []
        for i in range(test_length):
            predicted.append(series_train.mean())
        return None, predicted

    @staticmethod
    def naive_prediction(series_train, test_length, parameters_version=None):
        fitted_values = None
        predicted_one_val = series_train.last('1D')
        predicted = []
        for i in range(0, test_length):
            predicted.append(predicted_one_val[0])

        # print(series_train)
        # print(predicted)

        return fitted_values, predicted

    def _update_list_of_stations(self, data):
        print(self.stations)
        stations_used = [station for station in self.stations]
        for i, station in enumerate(self.stations):
            if len(data[station]) is 0:
                stations_used.remove(station)
        self.stations = set(stations_used)

    def save_data_to_file(self, file_name):
        output_file = open(file_name, 'w')
        writer = csv.writer(output_file, delimiter=',')
        for station in sorted(self.stations):
            if len(self.data[station]) > 0:
                writer.writerow(self.data[station])
        output_file.close()

    def save_dates_to_file(self, file_name, data_name):
        with open(file_name, "w") as dates_file:
            dates_writer = csv.writer(dates_file, delimiter=',')
            dates_writer.writerow(self.data_frame[data_name].Date)

    def save_stations_to_file(self, file_name):
        with open(file_name, "w") as stations_file:
            stations_writer = csv.writer(stations_file, delimiter=',')
            stations_writer.writerow(sorted(self.stations))
