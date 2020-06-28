import logging
import math
from collections import defaultdict

from data.FileWriter import FileWriter
from metrics.Statistics import Statistics
import matplotlib.pyplot as plt
import numpy as np


def data_statistics(title, data, stats=None, normalize=False, log_fun=logging.INFO):
    """
    Calculated data_statistics for Metrics.
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
    log_fun("Statistics %s" % title)

    all_data = []
    for key in data:
        if not isinstance(data[key], list):
            all_data.append(data[key])
        else:
            all_data.extend(data[key])

    if normalize:
        minimum = min(all_data)
        maximum = max(all_data)
        all_data = [(d - minimum) / (maximum - minimum) for d in all_data]

    result = Statistics.calculate(all_data, stats, log_fun)
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
        r_1 = min(data[key].values())
        r_2 = max(data[key].values())

        step = (r_2 - r_1) / n_bins if n_bins != -1 else 1
        bins = np.arange(start=r_1, stop=r_2 + 2 * step, step=step)

        cnt, bins = np.histogram(list(data[key].values()), bins)
        plt.plot(bins[:-1], cnt, label=key)

    plt.legend()


def category_histogram(title, data, category_data, labels, n_bins=10):
    """
    Plots data histogram.
    :param title: string
        Plot title
    :param data: dict
        Data for histogram
    :param n_bins: int
        Number of bins
    :param normalize: boolean
        True - values will be normalized
    """
    categorized_data = defaultdict(list)
    all_data = []
    for key in data:
        if not isinstance(data[key], list):
            categorized_data[category_data[key]].append(data[key])
            all_data.append(data[key])
        else:
            categorized_data[category_data[key]].extend(data[key])
            all_data.extend(data[key])

    fig, ax = plt.subplots()
    hist_data = [categorized_data[key] for key in sorted(categorized_data.keys())]

    bins = np.linspace(0, max(all_data), n_bins)
    # bins = np.linspace(0, 1, n_bins)

    colors = ['#7f7f7f', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    ax.hist(hist_data, bins, label=labels[len(labels)-len(hist_data):], color=colors[len(labels)-len(hist_data):])
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),  ncol=len(hist_data),
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
              title="Degree in (static)")
    plt.ylabel('Frequency')
    plt.xlabel("Metrics value")
    # plt.title(title)
    ax.set_xticks(bins)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def histogram(title, data, n_bins=10, half_open=False, integers=True, step=-1, normalize=False):
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
    bins = np.arange(start=r_1, stop=int(math.ceil(r_2 / step)) * step + step, step=step)
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
