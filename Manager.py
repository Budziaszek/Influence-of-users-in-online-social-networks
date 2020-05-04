import datetime as dt
import logging
import math
import os
import pickle
import warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import dendrogram

from statsmodels.tsa.statespace.tools import diff
from sklearn.cluster import KMeans, AgglomerativeClustering
from statistics import mean, stdev, variance
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pandas as pd

from DataProcessing.ResultsDisplay import histogram
from Metrics.Metrics import Metrics
from Metrics.MetricsProcessing.Histogram import Histogram
from Metrics.MetricsProcessing.Prediction import Prediction
from Metrics.config import degree_in_static
from Network.GraphIterator import GraphIterator
from Utility.Functions import make_data_positive, modify_data
from Utility.ProgressBar import ProgressBar
from DataProcessing.PostgresDatabaseEngine import PostgresDatabaseEngine
from Network.SocialNetworkGraph import SocialNetworkGraph
from DataProcessing.FileWriter import FileWriter
from Network.NeighborhoodMode import NeighborhoodMode


class Manager:
    _databaseEngine = PostgresDatabaseEngine()
    _number_of_days_in_interval = 7  # 7
    _number_of_new_days_in_interval = 3  # 3
    dynamic_graphs = []
    static_graph = None
    days = []
    comments_to_add = None
    neighborhood_mode = None
    colored = '\x1b[34m'
    background = '\x1b[30;44m'
    reset = '\x1b[0m'

    class CommentsReader:
        def __init__(self, method, include_responses_from_author):
            self._method = method
            self._include_responses_from_author = include_responses_from_author
            self._does_exist = False
            self._data = []

        def append_data(self, day_start, day_end):
            if not self._does_exist:
                self._data.append(self._method(day_start, day_end, self._include_responses_from_author))

        def get_data(self):
            self._does_exist = True
            return self._data

    class HistogramManager:
        def __init__(self, data_function, calculated_value, x_scale, size_scale):
            self.data_function = data_function
            if data_function is None:
                fun_name = "all"
            else:
                fun_name = str(data_function.__name__)
            self.file_name = fun_name + "_" + "hist_" + calculated_value.get_name() + ".txt"
            self._histogram = Histogram(x_scale, size_scale)

        def add_data(self, size, data):
            self._histogram.add(size, data, self.data_function)

        def save(self, folder):
            self._histogram.save(folder, self.file_name)

    def __init__(self, connection_parameters, test=False):
        """
        Creates DatabaseEngine that connects to database with given parameters and later allow operations
        (e.g. queries). Define time intervals. Initializes some variables for future use.
        :param connection_parameters: str
            Defines connection to database
        :param test: bool, optional
            True - calculate model only for short period of time
        """
        ProgressBar.set_colors(Manager.colored, Manager.reset)
        warnings.filterwarnings("ignore")
        self._comments_to_comments = self.CommentsReader(self._select_responses_to_comments, True)
        self._comments_to_posts = self.CommentsReader(self._select_responses_to_posts, True)
        self._comments_to_comments_from_others = self.CommentsReader(self._select_responses_to_comments, False)
        self._comments_to_posts_from_others = self.CommentsReader(self._select_responses_to_posts, False)

        self.histogram_managers = []
        self._databaseEngine.connect(connection_parameters)
        self._databaseEngine.create_first_activity_date_column()
        self._dates_range = self._get_dates_range()
        self._days_count = (self._dates_range[1] - self._dates_range[0]).days
        if test is True:
            self._days_count = self._number_of_days_in_interval * 5
        self.authors = self._get_authors_parameter("name")

    def calculate(self, metrics, condition_fun=None, log_fun=logging.INFO):
        """
        Calculated metrics for current neighborhood_mode (self.neighborhood_mode) and saves result to database.
        :param metrics: Metrics
            Metrics definition
        :param condition_fun: function
            Defines values that will be accepted (unaccepted will be replaced using Manager.modify_data)
        :param log_fun:
            Defines logging function.
        """
        log_fun("Calculating %s (%s)" % (metrics.get_name(), self.neighborhood_mode))
        first_activity_dates = self._get_first_activity_dates()
        if metrics.value in [Metrics.NEIGHBORHOOD_QUALITY]:
            users_selection = self.select_users(NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS,
                                                degree_in_static,
                                                percent=0.1)
            data = metrics.calculate(self.authors.keys(), first_activity_dates, users_selection=users_selection)
        else:
            data = metrics.calculate(self.authors.keys(), first_activity_dates)
        log_fun("Calculated " + metrics.get_name() + ". Saving...")
        log_fun("Calculated")

        bar = ProgressBar("Processing %s (%s)\n" % (metrics.get_name(), self.neighborhood_mode), "Processed",
                          len(self.authors.keys()))
        for i, user_id in enumerate(sorted(self.authors.keys())):  # For each author (node)
            bar.next()
            d = data[user_id]
            m = modify_data(d, condition_fun)
            self._save_to_database(self.neighborhood_mode.short() + "_" + metrics.get_name(), user_id, m)
        self._save_histograms_to_file(str(self.neighborhood_mode.value))
        bar.finish()
        log_fun("Saved " + metrics.get_name() + ".")

    def clean(self, neighborhood_mode, metrics):
        """
        Drops result column from database.
        :param neighborhood_mode: NeighborhoodMode
            Neighborhood mode definition
        :param metrics: Metrics
            Metrics definition
        """
        self._databaseEngine.drop_column(neighborhood_mode.short() + "_" + metrics.get_name())

    def get_data(self, neighborhood_mode, metrics, cut_down=float("-inf"), cut_up=float("inf"), users_selection=None):
        """
         :param neighborhood_mode: NeighborhoodMode
            Neighborhood mode definition
        :param metrics: Metrics
            Metrics definition
        :param cut_down:
            Minimum accepted value
        :param cut_up:
            Maximum accepted value
        :param users_selection:
            Accepted users ids
        :return: dict
            Dictionary of ids and metrics values.
        """
        # Get data
        data = self._databaseEngine.get_array_value_column(neighborhood_mode.short() + "_" + metrics.get_name())
        users_selection = data.keys() if users_selection is None else users_selection
        # Cut data
        return {k: v for k, v in data.items() if cut_down < v < cut_up and k in users_selection}

    def correlation(self, neighborhood_mode, metrics, functions, do_abs=True, log_fun=logging.INFO):
        """
        Calculates, saves and displays correlation.
        :param neighborhood_mode: NeighborhoodMode
            Neighborhood mode definition
        :param metrics: iterable
            Metrics definitions
        :param functions: iterable
            Functions which should be used in case of dynamic graphs.
        :param do_abs: boolean
            True - call abs function on correlation values.
        :param log_fun: function
            Defines logging function
        """
        pd.set_option('display.width', None)
        keys = sorted(self.authors.keys())
        df = pd.DataFrame()
        for m in metrics:
            if m.graph_iterator.graph_mode[0] is GraphIterator.ITERATOR.DYNAMIC:
                for f in functions:
                    data = self._databaseEngine.get_array_value_column(neighborhood_mode.short() + "_"
                                                                       + m.get_name(), f)
                    x = []
                    for key in keys:
                        x.append(data[key])
                    df[m.get_name() + "_" + f.__name__] = x
            else:
                data = self._databaseEngine.get_array_value_column(neighborhood_mode.short() + "_" + m.get_name())
                x = []
                for key in keys:
                    x.append(data[key])
                df[m.get_name()] = x

        if not os.path.exists('output/correlation'):
            os.mkdir('output/correlation')

        c = df.corr()

        f = plt.figure()
        plt.matshow(c, fignum=f.number)
        plt.yticks(range(df.shape[1]), df.columns, fontsize=6)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=10)
        plt.title('Correlation Matrix', fontsize=16)

        c.to_csv(r'output/correlation/' + 'correlation' + '.txt', index=True, header=True)

        c.style.background_gradient(cmap='Blues')

        if do_abs:
            c.abs()

        c_rate = {}
        for row in c:
            c_rate[row] = {}
            c_rate[row]['extreme'] = []
            c_rate[row]['very high'] = []
            c_rate[row]['high'] = []
            c_rate[row]['medium'] = []
            c_rate[row]['small'] = []
            c_rate[row]['very small'] = []

        for row in c:
            for column in c:
                if row != column:
                    if c[row][column] > 0.9:
                        c_rate[row]['extreme'].append(column)
                    elif c[row][column] > 0.8:
                        c_rate[row]['very high'].append(column)
                    elif c[row][column] > 0.7:
                        c_rate[row]['high'].append(column)
                    elif c[row][column] > 0.6:
                        c_rate[row]['medium'].append(column)
                    elif c[row][column] > 0.5:
                        c_rate[row]['small'].append(column)
                    elif c[row][column] is not np.nan:
                        c_rate[row]['very small'].append(column)

        for key in c_rate:
            log_fun(key)
            for r in c_rate[key]:
                log_fun("\t", str(r), str(c_rate[key][r]))

        plt.show()

    def table(self, neighborhood_mode, metrics, functions, title='table', table_mode="index"):
        """
        Creates table of metrics values for all users.
        :param neighborhood_mode: NeighborhoodMode
            Neighborhood mode definition
        :param metrics: iterable
            Metrics definitions
        :param functions: iterable
            Functions which should be used in case of dynamic graphs.
        :param title: string
            File title (ending)
        :param table_mode: string
            Defines table values. Included in file title (beelining)
            'index' - index in table
            'value' - normal metrics value
            'name' - user name ordered by index table
        :return:
        """
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)
        keys = sorted(self.authors.keys())
        df = pd.DataFrame()
        df['Name'] = [self.authors[key] for key in keys]
        for m in metrics:
            # logging.INFO("Reading.. " + m.get_name())
            if m.graph_iterator.graph_mode[0] is GraphIterator.ITERATOR.DYNAMIC:
                for f in functions:
                    data = self._databaseEngine.get_array_value_column(neighborhood_mode.short() + "_" + m.get_name(),
                                                                       f)
                    x = []
                    for key in keys:
                        x.append(data[key])
                    df[m.get_name() + "_" + f.__name__] = x
                    if table_mode is not "value":
                        df[m.get_name() + "_" + f.__name__] = df[m.get_name() + "_" + f.__name__].rank(ascending=False)
            else:
                data = self._databaseEngine.get_array_value_column(neighborhood_mode.short() + "_" + m.get_name())
                x = []
                for key in keys:
                    x.append(data[key])
                df[m.get_name()] = x
                if table_mode is not "value":
                    df[m.get_name()] = df[m.get_name()].rank(ascending=False)
        if table_mode is not "name":
            result = df
        else:
            metrics_keys = df.keys()
            authors_rankings = pd.DataFrame()
            for m in metrics_keys:
                logging.INFO("Checking table for.. " + m)
                if m != 'Name':
                    authors_rankings[m] = list(df.sort_values(m, ascending=False)['Name'])
            result = authors_rankings

        if not os.path.exists('output/table'):
            os.mkdir('output/table')

        result.to_csv(r'output/table/' + table_mode + "_" + title + '.txt', index=True, header=True)

    def display_between_range(self, neighborhood_mode, metrics, minimum, maximum, log_fun=logging.INFO):
        """
        Displays users for whom metrics value is between range (minimum, maximum).
        :param neighborhood_mode: NeighborhoodMode
            Neighborhood mode definition
        :param metrics: Metrics
            Metrics definition
        :param minimum: int
        :param maximum: int
        :param log_fun: function
            Defines logging function
        """
        if metrics is not None:
            data = self._databaseEngine.get_array_value_column(neighborhood_mode.short() + "_" + metrics.get_name())
            for item in sorted(data.items(), key=lambda k: (k[1], k[0]), reverse=True):
                if minimum < item[1] < maximum:
                    log_fun(str(self.authors[item[0]]) + " " + str(item[1]))

    def select_users(self, neighborhood_mode, metrics, values_start=None, values_stop=None, percent=None):
        data = self._databaseEngine.get_array_value_column(neighborhood_mode.short() + "_" + metrics.get_name())
        sorted_keys = [item[0] for item in sorted(data.items(), key=lambda k: (k[1], k[0]))]
        sorted_keys.reverse()
        if percent is None:
            return [k for k in sorted_keys if values_start <= data[k] <= values_stop]
        else:
            return sorted_keys[0:math.ceil(len(sorted_keys) * percent)]

    def k_means(self, n_clusters, parameters, users_selection=None, log_fun=logging.info):
        """
        Performs k-means clustering and displays results.
        :param n_clusters: int
            Number of clusters.
        :param parameters: array (string)
            Parameters included in clustering.
         :param users_selection:
            Cluster only some selected users.
        :param log_fun: function
            Defines logging function
        """
        log_fun('Clustering: k-means, n_clusters= %s...' % str(n_clusters) + '\n')
        data, _data = self._prepare_clustering_data(parameters, users_selection)
        log_fun('\tData prepared.')
        k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
        k_means.fit(data)
        log_fun('\tData fitted.')
        self._save_clusters([p[0].short() + "_" + p[1].get_name() for p in parameters],
                            k_means.labels_, n_clusters, _data, users_selection, log_fun=log_fun)
        log_fun('Clustering finished. Result saved in output folder.')

    def plot_dendrogram(self, model, **kwargs):
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                          counts]).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)

    def agglomerative_clustering(self, n_clusters, parameters, users_selection=None, log_fun=logging.info):
        log_fun('Clustering: agglomerative clustering...' + '\n')
        data, _data = self._prepare_clustering_data(parameters, users_selection)
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(data)

        print('done')
        # self.plot_dendrogram(clustering, truncate_mode='level', p=3)
        # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        # plt.show()

        self._save_clusters([p[0].short() + "_" + p[1].get_name() for p in parameters],
                            clustering.labels_, n_clusters, _data, users_selection, log_fun=log_fun)
        log_fun('Clustering finished. Result saved in output folder.')

    def predict(self, data, author_id):
        """
        Predict time series.
        :param data: array
            DataProcessing used in prediction
        :param author_id: int
            Author id
        """
        plot_data = []
        title_data = []
        interesting_ids = [1672, 440, 241, 2177, 797, 3621, 11, 2516]
        # methods = [Prediction.exponential_smoothing, Prediction.ARIMA]
        parameters_versions = [i for i in range(3)]
        data = make_data_positive(diff(data))
        logging.info("Predict")
        prediction = Prediction(data, self.authors[author_id])
        for parameters_version in parameters_versions:
            result = prediction.predict(0, len(data) - 50, 50, Prediction.exponential_smoothing,
                                        Prediction.MAPE_error, parameters_version)
            if len(plot_data) == 0:
                plot_data.append((result[2].index, result[2]))
                title_data.append("Original")
            plot_data.append(result[1])
            title_data.append(result[0] + str(parameters_version))
        if self.authors[author_id] in interesting_ids:
            Prediction.plot("Prediction for " + self.authors[author_id], plot_data, title_data)

    @staticmethod
    def calculate_trend(data):
        y = np.array(data)
        n = np.size(data)
        x = np.array([i for i in range(len(data))])
        m_x, m_y = np.mean(x), np.mean(y)
        ss_xy = np.sum(y * x) - n * m_y * m_x
        ss_xx = np.sum(x * x) - n * m_x * m_x
        a = ss_xy / ss_xx
        b = m_y - a * m_x
        return b, a

    def stability(self, neighborhood_mode, metrics, users_selection=None):
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)
        data = self._databaseEngine.get_array_value_column(neighborhood_mode.short() + "_" + metrics.get_name(), 'all')
        keys = sorted(data.keys(), reverse=True) if not users_selection else users_selection

        stability = pd.DataFrame()
        stability['id'] = keys
        stability['len'] = [len(data[key]) for key in keys]
        stability['mean'] = [mean(data[key]) for key in keys]
        stability['variance'] = [variance(data[key]) if len(data[key]) > 1 else np.nan for key in keys]
        stability['std'] = [np.std(data[key]) if len(data[key]) > 1 else np.nan for key in keys]

        data_moving_average = defaultdict(list)

        N = 50
        for key in keys:
            data_moving_average[key] = list(pd.Series(data[key]).rolling(window=N).mean().iloc[N - 1:].values)

        stability['trend'] = [Manager.calculate_trend(data[key])[1] for key in keys]

        trend_dic = {}
        c = 0
        for key in keys:
            trend_arr = []
            M = math.floor(N / 2)
            k = math.ceil(len(data[key]) / M)
            if c < 5:
                fig, ax = plt.subplots()
                ax.plot(data[key])
                b, a = Manager.calculate_trend(data[key])
                ax.plot([b + a * t_i for t_i in range(len(data[key]))], color='orange')
                fig.suptitle(self.authors[key])
            for i in range(k):
                start = i * M
                end = (i + 1) * M if (i + 1) * M < len(data[key]) else len(data[key])
                b, a = Manager.calculate_trend(data[key][start:end])
                if not math.isnan(a):
                    trend_arr.append(a)
                if c < 5:
                    ax.plot([x for x in range(start, end)], [b + a * t_i for t_i in range(len(data[key][start:end]))],
                            color='red')
            trend_dic[key] = trend_arr
            c += 1

        stability['trend_min'] = [min(trend_dic[key]) for key in keys]
        stability['trend_max'] = [max(trend_dic[key]) for key in keys]
        stability['trend_mean'] = [mean(trend_dic[key]) for key in keys]
        stability['trend_sum'] = [sum(trend_dic[key]) for key in keys]

        adf = []
        # for key in keys:
        #     try:
        #         adf.append(adfuller(data[key])[0])
        #     except Exception as e:
        #         adf.append(np.nan)
        #         logging.WARNING(str(e))
        # stability['adf'] = adf


        FileWriter.write_data_frame_to_file(FileWriter.STABILITY, metrics.get_name(), stability)

        histogram("Stability std", {key: stability.set_index('id')['std'][key] for key in keys}, n_bins=15,
                  integers=False)
        histogram("Stability trend", {key: stability.set_index('id')['trend'][key] for key in keys}, n_bins=15,
                  integers=False)
        histogram("Stability trend max", {key: stability.set_index('id')['trend_max'][key] for key in keys}, n_bins=15,
                  integers=False)

        plt.show()

    def _get_dates_range(self):
        """
        Checks dates range (find min, max action (comment or post) time and number of days)
        :return: tuple
            Contains first and last date occurring in database
        """
        r = []
        for column in ["comments", "posts"]:
            r.append(self._databaseEngine.get_dates_range(column))
        dates_range = (min(r[0][0], r[1][0]), max(r[0][1], r[1][1]))
        return dates_range

    def _select_responses_to_posts(self, day_start, day_end, include_responses_from_author):
        """
        Executes query, which selects comments added to posts.
        :param day_start: datetime.datetime
        :param day_end: datetime.datetime
        :param include_responses_from_author: bool
            True - reactions from author should be included
        """
        c = self._databaseEngine.get_responses_to_posts(day_start, day_end, include_responses_from_author)
        return np.array(c)

    def _select_responses_to_comments(self, day_start, day_end, include_responses_from_author):
        """
        Executes query, which selects comments added to comments.
        :param day_start: datetime.datetime
        :param day_end: datetime.datetime
        :param include_responses_from_author: bool
            True - reactions from author should be included
        """
        c = self._databaseEngine.get_responses_to_comments(day_start, day_end, include_responses_from_author)
        return np.array(c)

    def _select_comments(self, day_start, day_end):
        """
        Calls methods, which selects all required comments if they weren't already selected.
        :param day_start: datetime.datetime
        :param day_end: datetime.datetime
        """
        if self.neighborhood_mode.do_read_comments_to_posts:
            self._comments_to_posts.append_data(day_start, day_end)
        if self.neighborhood_mode.do_read_comments_to_comments:
            self._comments_to_comments.append_data(day_start, day_end)
        if self.neighborhood_mode.do_read_comments_to_posts_from_others:
            self._comments_to_posts_from_others.append_data(day_start, day_end)
        if self.neighborhood_mode.do_read_comments_to_comments_from_others:
            self._comments_to_comments_from_others.append_data(day_start, day_end)

    def _set_comments_to_add(self):
        """
        Sets variable, which contains comments included in model.
        """
        self.comments_to_add = []
        if self.neighborhood_mode is NeighborhoodMode.COMMENTS_TO_POSTS:
            self.comments_to_add.append(self._comments_to_posts.get_data())
        if self.neighborhood_mode is NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS:
            self.comments_to_add.append(self._comments_to_posts_from_others.get_data())
        if self.neighborhood_mode is NeighborhoodMode.COMMENTS_TO_COMMENTS:
            self.comments_to_add.append(self._comments_to_comments.get_data())
        if self.neighborhood_mode is NeighborhoodMode.COMMENTS_TO_COMMENTS_FROM_OTHERS:
            self.comments_to_add.append(self._comments_to_comments_from_others.get_data())
        if self.neighborhood_mode is NeighborhoodMode.COMMENTS_TO_POSTS_AND_COMMENTS:
            self.comments_to_add.append(self._comments_to_posts.get_data())
            self.comments_to_add.append(self._comments_to_comments.get_data())
        if self.neighborhood_mode is NeighborhoodMode.COMMENTS_TO_POSTS_AND_COMMENTS_FROM_OTHERS:
            self.comments_to_add.append(self._comments_to_posts_from_others.get_data())
            self.comments_to_add.append(self._comments_to_comments_from_others.get_data())

    def _read_salon24_comments_data_by_day(self):
        """
        Retrieves the most important values about comments (tuple: (comment author, post author)) by day from database
        and store values in array (chronologically day by day)
        """
        days = []
        bar = ProgressBar("Selecting _data", "DataProcessing selected", self._days_count)
        for day in (self._dates_range[0] + dt.timedelta(n) for n in range(self._days_count)):
            day_start = day.replace(hour=00, minute=00, second=00)
            day_end = day.replace(hour=23, minute=59, second=59)
            days.append(day_start.date())
            self._select_comments(day_start, day_end)
            bar.next()
        bar.finish()
        self._set_comments_to_add()
        self.days = days

    def _add_data_to_graphs(self, graph_type):
        """
        Adds selected comments data to graphs.
        :param graph_type: string
            Defines whenever static graph or dynamics graphs should be created: "s" - static, "d" - dynamic, "sd" - both
        """
        self.dynamic_graphs = []
        self.static_graph = None
        if graph_type is "sd":
            logging.info("Creating static graph and dynamic graphs")
            self._add_data_to_static_graph()
            self._add_data_to_dynamic_graphs()
        elif graph_type is "d":
            logging.info("Creating dynamic graphs")
            self._add_data_to_dynamic_graphs()
        elif graph_type is "s":
            logging.info("Creating static graph")
            self._add_data_to_static_graph()
        else:
            raise Exception("EXCEPTION - wrong graph value")
        logging.info("Graphs created")

    def _add_data_to_static_graph(self):
        """
        Adds data to static graph - all days.
        """
        graph = SocialNetworkGraph()
        for comments in self.comments_to_add:
            edges = np.concatenate(comments, axis=0)
            graph.add_edges(edges)
        graph.start_day, graph.end_day = self.days[0], self.days[-1]
        self.static_graph = graph

    def _add_data_to_dynamic_graphs(self):
        """
        Adds data to dynamic graphs - selected days only.
        """
        graph = SocialNetworkGraph()
        step = self._number_of_new_days_in_interval
        interval_length = self._number_of_days_in_interval
        i = 0
        while i + step <= self._days_count:
            end = i + interval_length - 1
            if end >= self._days_count:
                return
            for comments in self.comments_to_add:
                edges = np.concatenate(comments[i:end + 1], axis=0)
                graph.add_edges(edges)
            graph.start_day, graph.end_day = self.days[i], self.days[end]
            self.dynamic_graphs.append(graph)
            graph = SocialNetworkGraph()
            i += step

    def check_graphs(self, neighborhood_mode):
        if self.static_graph is None or self.dynamic_graphs is None:
            self._create_graphs(neighborhood_mode)

    def _create_graphs(self, neighborhood_mode):
        """
        Creates graphs corresponding to the selected neighborhood_mode.
        :param neighborhood_mode: NeighborhoodMode
            Defines model neighborhood_mode (which comments should me included)
        """
        self.neighborhood_mode = neighborhood_mode
        graphs_file_name = 'graphs' + "/" + self.neighborhood_mode.name \
                           + "_" + str(self._number_of_days_in_interval) \
                           + "_" + str(self._number_of_new_days_in_interval)
        Path('graphs').mkdir(parents=True, exist_ok=True)

        if os.path.exists(graphs_file_name):
            self._load_graphs_from_file(graphs_file_name)
            GraphIterator.set_graphs(self.static_graph, self.dynamic_graphs)
        else:
            self._read_salon24_comments_data_by_day()
            self._add_data_to_graphs("sd")
            GraphIterator.set_graphs(self.static_graph, self.dynamic_graphs)
            self._save_graphs_to_file(graphs_file_name)

    def _load_graphs_from_file(self, graphs_file_name):
        """
        Loads graphs from file.
        :param graphs_file_name:
            Filename from which graphs will be loaded
        """
        logging.info("Loading graphs from file")
        with open(graphs_file_name, 'rb') as file:
            dictionary = pickle.load(file)
            self.static_graph = dictionary['static']
            self.dynamic_graphs = dictionary['dynamic']

    def _prepare_clustering_data(self, parameters, users_selection=None):
        """
        Prepares data about parameters names in a form of an array, which can be used in clustering.
        :param parameters: array (string)
            Parameters included in clustering.
        :return: numpy.array
            Data for clustering.
        """
        data = {}
        _data = {}

        for parameter in parameters:
            neighborhood_mode = parameter[0]
            metrics = parameter[1]
            weight = parameter[2]
            f = None if len(parameter) == 3 else parameter[3]
            name = neighborhood_mode.short() + "_" + metrics.get_name()
            r = self._databaseEngine.get_array_value_column(name, f)
            users_selection = self.authors.keys() if not users_selection else users_selection
            data[name] = [r[k] for k in sorted(r.keys()) if k in users_selection]
            minimum, maximum = min(data[name]), max(data[name])
            _data[name] = data[name].copy()
            data[name] = [(x - minimum) / (maximum - minimum) * weight for x in data[name]]
        return np.column_stack(data[p[0].short() + "_" + p[1].get_name()] for p in parameters), np.column_stack(
            _data[p[0].short() + "_" + p[1].get_name()] for p in parameters)

    class Cluster:
        def __init__(self):
            pass

    def _save_clusters(self, parameters_names, classes, n_clusters, data, users_selection=None, log_fun=logging.debug):
        """
        Displays results of clustering.
        :param parameters_names: array (string)
            Parameters included in clustering.
        :param classes: array
            Classes created
        :param n_clusters: int
            Number of clusters
        """
        ranges = defaultdict(list)
        p_values = defaultdict(list)
        save = defaultdict(list)

        interesting_users_50 = self._databaseEngine.get_interesting_users(50)
        interesting_users_250 = self._databaseEngine.get_interesting_users(250)
        interesting_users_500 = self._databaseEngine.get_interesting_users(500)
        interesting_users_1000 = self._databaseEngine.get_interesting_users(1000)

        for cluster in range(n_clusters):
            indexes = np.where(classes == cluster)
            users_ids = np.array(sorted(self.authors.keys()))[indexes] if not users_selection \
                else np.array(sorted(users_selection))[indexes]

            log_fun('Cluster: %s' % cluster)
            # log_fun(
            #     '\t center: %s\n\t number of users: %s' % ([round(c, 3) for c in n_clusters[cluster]], len(users_ids)))
            log_fun('Parameters:')

            save['stats'].extend(['min', 'max', 'mean', 'stdev'])
            empty_stats = ['' for _ in range(3)]
            save['cluster'].append(cluster)
            save['cluster'].extend(empty_stats)
            save['size'].append(len(users_ids))
            save['size'].extend(empty_stats)

            for i, parameter_name in enumerate(parameters_names):
                values = data[indexes, i][0]
                r = [round(min(values), 3),
                     round(max(values), 3),
                     round(mean(values), 3),
                     round(stdev(values) if len(values) > 1 else values[0], 3)]
                ranges[parameter_name].append([r[0], r[1]])
                p_values[parameter_name].append(values)
                save[parameter_name].extend(r)
                log_fun("\t %s: min= %s, max= %s, mean= %s, stdev= %s" % (parameter_name, r[0], r[1], r[2], r[3]))

            log_fun("Sample users:")
            s = []
            len_50 = 0
            len_250 = 0
            len_500 = 0
            len_1000 = 0

            for i in users_ids:
                if i in interesting_users_50:
                    s.append(self.authors[i])
                    parameters = [
                        self._databaseEngine.get_array_value_column_for_user(parameter_name, i, mean)
                        for parameter_name in parameters_names]
                    log_fun("\t %s: %s" % (self.authors[i], [round(p, 3) for p in parameters]))
                    len_50 += 1
                    len_250 += 1
                    len_500 += 1
                    len_1000 += 1
                elif i in interesting_users_250:
                    len_250 += 1
                    len_500 += 1
                    len_1000 += 1
                elif i in interesting_users_500:
                    len_500 += 1
                    len_1000 += 1
                elif i in interesting_users_1000:
                    len_1000 += 1
            log_fun('\n')
            save['sample'].extend([s, *empty_stats])
            save['first 50'].extend([len_50, len_50 / len(users_ids) * 100, *empty_stats[:-1]])
            save['first 250'].extend([len_250, len_250 / len(users_ids) * 100, *empty_stats[:-1]])
            save['first 500'].extend([len_500, len_500 / len(users_ids) * 100, *empty_stats[:-1]])
            save['first 1000'].extend([len_1000, len_1000 / len(users_ids) * 100, *empty_stats[:-1]])
        FileWriter.write_split_row_to_file(FileWriter.CLUSTERING, str(n_clusters), ['cluster', save['cluster']])
        FileWriter.write_split_row_to_file(FileWriter.CLUSTERING, str(n_clusters), ['size', save['size']])
        FileWriter.write_split_row_to_file(FileWriter.CLUSTERING, str(n_clusters), ['stats', save['stats']])
        for parameter_name in parameters_names:
            FileWriter.write_split_row_to_file(FileWriter.CLUSTERING, str(n_clusters), [parameter_name,
                                                                                             save[parameter_name]])
        FileWriter.write_split_row_to_file(FileWriter.CLUSTERING, str(n_clusters), ['sample', save['sample']])
        FileWriter.write_split_row_to_file(FileWriter.CLUSTERING, str(n_clusters), ['first 50', save['first 50']])
        FileWriter.write_split_row_to_file(FileWriter.CLUSTERING, str(n_clusters), ['first 250', save['first 250']])
        FileWriter.write_split_row_to_file(FileWriter.CLUSTERING, str(n_clusters), ['first 500', save['first 500']])
        FileWriter.write_split_row_to_file(FileWriter.CLUSTERING, str(n_clusters), ['first 1000', save['first 1000']])

        Manager.plot_overlapping(parameters_names, ranges, False)
        Manager.plot_overlapping(parameters_names, p_values, True)

        plt.show()

    @staticmethod
    def plot_overlapping(parameters_names, arr, points=False):
        fig, axs = plt.subplots(len(parameters_names))
        plots = None
        for i, key in enumerate(parameters_names):
            plots = Manager.plot_overlapping_ranges_by_points(arr[key], axs[i]) if points \
                else Manager.plot_overlapping_ranges(arr[key], axs[i])
            axs[i].set_title(key, fontsize=6)
        fig.legend(plots, labels=['Cluster ' + str(i) for i in range(len(parameters_names))], loc="center right",
                   bbox_to_anchor=(0.99, 0.5), borderaxespad=0.1, prop={'size': 6})
        fig.tight_layout()
        plt.subplots_adjust(right=0.85)

    @staticmethod
    def plot_overlapping_ranges_by_points(points, axs):
        axs.set_ylim([-2, 2 * len(points) + 2])
        axs.yaxis.set_visible(False)

        for item in axs.get_xticklabels():
            item.set_fontsize(6)

        x = points
        y = [i * 2 for i in range(len(points))]

        plots = []
        for i in range(len(points)):
            plot = axs.plot(x[i], [y[i] for _ in range(len(x[i]))], 'o')
            for p in plot:
                p.set_alpha(0.5)
            plots.append(plot)

        return plots

    @staticmethod
    def plot_overlapping_ranges(ranges, axs):
        axs.yaxis.set_visible(False)

        for item in axs.get_xticklabels():
            item.set_fontsize(6)

        width = [(r[1] - r[0]) for r in ranges]
        center = [r[0] + (r[1] - r[0]) / 2 for r in ranges]
        height = [1 for _ in range(len(ranges))]
        bottom = [i for i in range(len(ranges))]

        bars = []
        for i in range(len(ranges)):
            bar = axs.bar(center[i], height[i], width=width[i], bottom=bottom[i])
            for b in bar:
                b.set_alpha(0.5)
            bars.append(bar)

        return bars

    def _save_data_to_file(self, file_writer, author_id, data):
        """
        Saves data to file.
        :param file_writer: FileWriter
            FileWriter used to save data
        :param author_id: int
            Author id
        :param data: array (float)
            Author's data
        """
        if file_writer is not None:
            file_writer.write_row_to_file([author_id, self.authors[author_id], *data])

    def _add_data_to_histograms(self, size, data):
        """
        Adds data to histograms.
        :param size: array (int)
            Neighborhood size
        :param data:
            DataProcessing to add
        """
        for h in self.histogram_managers:
            h.add_data(size, data)

    def _initialize_histogram_managers(self, calculate_histogram, data_functions, value, x_scale, size_scale):
        """
        Initializes array of HistogramManagers if required.
        :param calculate_histogram: bool
            True - initialization required
        :param data_functions: array (function)
            Function used for data aggregation
        :param value: MetricsType
            Metrics that is currently calculated
        :param x_scale: array (float)
            Defines standard classes for the histogram
        :param size_scale: array (int)
            Defines size classes for the histogram
        :return: array (HistogramManager)
            Array of initialized HistogramManagers if initialization is required otherwise empty array
        """
        self.histogram_managers = []
        if calculate_histogram:
            # TODO fix
            self._calculate_and_get_authors_static_neighborhood_size()
            for data_function in data_functions:
                self.histogram_managers.append(self.HistogramManager(data_function, value, x_scale, size_scale))

    def _get_authors_parameter(self, parameter):
        """
        Created array of authors parameters ordered by id.
        :param parameter: str
            Parameter which should be selected from database
        :return: array
            Array with id and authors parameters
        """
        return self._databaseEngine.get_authors_parameter(parameter)

    def _get_graphs_labels(self, empty_count=0):
        """
        Created array of graphs labels.
        :param empty_count:
            Number of empty columns (without labels)
        :return: array (str)
            Graphs labels
        """
        row_captions = [str(g.start_day) for g in self.dynamic_graphs]
        row_captions[0] = "static"
        for e in range(empty_count):
            row_captions.insert(0, ",")
        return row_captions

    def _get_first_activity_dates(self):
        """
        Gets first_activity_date column.
        :return: dict
        """
        return dict(self._databaseEngine.get_first_activity_date())

        # def _get_users_first_activity_date(self, author_id):
        #     """
        #     Checks author's first activity date and returns it.
        #     :param author_id:
        #         Id of author
        #     :return: datetime.datetime
        #         First activity date
        #     """
        #     try:
        #         return self._databaseEngine.execute("SELECT %s FROM authors WHERE id = %s"
        #                                             % ("first_activity_date", author_id))[0][0]
        #     except IndexError:
        #         return None

        # def _calculate_and_get_authors_static_neighborhood_size(self):
        #     """
        #     Calculates static neighborhood size and returns it.
        #     :return: array (int)
        #         Neighborhoods sizes
        #     """
        #     if self.static_neighborhood_size is None:
        #         calculated_value = Metrics(Metrics.DEGREE_CENTRALITY, GraphConnectionType.IN,
        #                                    GraphIterator(GraphIterator.GraphMode.STATIC))
        #         self.static_neighborhood_size = {i: calculated_value.calculate(i, self._get_users_first_activity_date(i))[0]
        #                                          for i in self.authors.keys()}
        #     return self.static_neighborhood_size

    def _save_to_database(self, column_name, author_id, data):
        """
        Saves data for one author to database.
        :param author_id: int
            Id of the author to whom the data refer
        :param data: array (float)
            Calculated metrics values
        """
        self._databaseEngine.update_array_value_column(column_name, author_id, data)

    def _save_graphs_to_file(self, graphs_file_name):
        """
        Saves graphs to file.
        :param graphs_file_name: string
            Filename in which graphs will be saved
        """
        with open(graphs_file_name, 'wb') as file:
            pickle.dump({'static': self.static_graph, 'dynamic': self.dynamic_graphs}, file)

    def _save_histograms_to_file(self, mode_name):
        """
        Saves histogram data to file.
        :param mode_name: str
            NeighborhoodMode name - folder
        """
        for histogram in self.histogram_managers:
            histogram.save('output/' + mode_name + "/")
