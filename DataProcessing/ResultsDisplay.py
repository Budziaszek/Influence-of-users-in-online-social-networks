import logging
import math
from itertools import chain

from DataProcessing.FileWriter import FileWriter
from Metrics.MetricsProcessing.Statistics import Statistics
import matplotlib.pyplot as plt
import numpy as np


def data_statistics(title, data, stats=None, normalize=False, log_fun=logging.INFO):
    """
    Calculated data_statistics for metrics.
    :param title: string
        Title
    :param data: dict
        Data
    :param stats: iterable
        Functions to be calculated (if None default functions set will be used)
    :param normalize: boolean
        True - values will be normalized
    :param log_fun: function
        Defines logging function.
    """
    log_fun("Statistics %s (%s)" % title)

    if isinstance(data[list(data.keys())[-1]], list):
        data = list(chain(*data.values()))

    if normalize:
        minimum = min(data)
        maximum = max(data)
        data = [(d - minimum) / (maximum - minimum) for d in data]
    result = Statistics.calculate(list(data), stats)
    FileWriter.write_dict_to_file(FileWriter.STATISTICS, title + ".txt", result)


def points_2d(data_x, data_y, title_x, title_y):
    plt.figure()

    d = dict([(k, [data_x[k], data_y[k]]) for k in data_x])

    plt.plot([d[k][0] for k in d.keys()], [d[k][1] for k in d.keys()], '.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlabel(title_x)
    plt.ylabel(title_y)


def distribution_linear(data, n_bins=-1):
    """
    Plots line chart representing variables distribution.
    :param data: dict
        Data in dicts (dict od dicts)
    :param n_bins: int
        Number of bins
    """
    plt.figure()
    plt.rc('font', size=10)
    plt.ylabel('Frequency')
    plt.xlabel("Metrics value")
    for key in data:

        r_1 = min(data.values())
        r_2 = max(data.values())

        step = (r_2 - r_1) / n_bins if n_bins != -1 else 1
        bins = np.arange(start=r_1, stop=r_2 + 2 * step, step=step)

        cnt, bins = np.histogram(list(data[key].values()), bins)
        plt.plot(bins[:-1], cnt, label=key)

    plt.legend()


# TODO multiple bars
def histogram(title, data, n_bins, half_open=False, integers=True, step=-1, normalize=False):
    """
    Plots data histogram.
    :param title: string
        Plot title
    :param data: dict
        Data for histogram
    :param n_bins: int
        Number of bins
    :param half_open: boolean
        True - leaves last bin half-opened
    :param integers: boolean
        True - labels are considered as integers
    :param step: float
        Alternative for n_bins. Defines step for bins.
    :param normalize: boolean
        True - values will be normalized
    """

    def format_bin_label(value, i):
        return str(int(value)) if i else "{:.3f}".format(value)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.rc('font', size=8)
    fig, ax = plt.subplots()

    if normalize:
        minimum = min(data.values())
        maximum = max(data.values())
        data = {k: (data[k] - minimum) / (maximum - minimum) for k in data}

    r_1 = min(data.values())
    r_2 = max(data.values())

    step = (r_2 - r_1) / n_bins if step is -1 else step
    if integers:
        step = math.ceil(step)
    bins = np.arange(start=r_1, stop=r_2 + step, step=step)
    if half_open:
        bins = np.append(bins[:-1], [r_2])

    cnt, bins = np.histogram(list(data.values()), bins)
    if len(bins) > 1:
        labels = ["[" + format_bin_label(bins[i], integers) + ", "
                  + format_bin_label(bins[i + 1], integers) + ")" for i in range(len(bins) - 2)]
        last_sign = ")" if half_open else "]"
        labels.append("[" + format_bin_label(bins[-2], integers) + ", "
                      + format_bin_label(bins[-1], integers) + last_sign)
    else:
        labels = ["[" + str(bins[-2]) + ", " + str(bins[-1]) + "]"]

    x = np.arange(len(labels))  # the label locations
    width = 0.9  # the width of the bars

    rects1 = ax.bar(x, cnt, width)

    plt.ylabel('Frequency')
    plt.xlabel("Metrics value")
    plt.title(title)
    ax.set_xticklabels(bins)
    ax.set_xticklabels(labels)
    ax.set_xticks(x)
    plt.xticks(rotation=90)
    plt.ylim(0, max(cnt) * 1.1)
    autolabel(rects1)
    fig.tight_layout()


def show_plots():
    """Displays plots which were prepared"""
    plt.show()

