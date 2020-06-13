import datetime as dt
import logging
import math
import os
import pickle
import sys
import time
import warnings
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import dendrogram
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.tree import DecisionTreeClassifier

from statsmodels.tsa.statespace.tools import diff
from sklearn.cluster import KMeans, AgglomerativeClustering
from statistics import mean, stdev, variance
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pandas as pd

from DataProcessing.ResultsDisplay import histogram
from Metrics.Metrics import Metrics
from Metrics.MetricsProcessing.Prediction import Prediction
from Metrics.config import degree_in_static
from Network.GraphIterator import GraphIterator
from Utility.Functions import make_data_positive, modify_data, sum_by_key, fun_all
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
    posts_counts = []
    responses_counts = []
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
        self._databaseEngine.create_activity_date_columns()
        self._dates_range = self._get_dates_range()
        self._days_count = (self._dates_range[1] - self._dates_range[0]).days
        if test is True:
            self._days_count = self._number_of_days_in_interval * 5
        self.authors = self._get_authors_parameter("name")

    def calculate(self, metrics, condition_fun=None, log_fun=logging.info):
        """
        Calculated metrics for current neighborhood_mode (self.neighborhood_mode) and saves result to database.
        :param metrics: Metrics
            Metrics definition
        :param condition_fun: function
            Defines values that will be accepted (unaccepted will be replaced using Manager.modify_data)
        :param log_fun:
            Defines logging function.
        """
        try:
            log_fun("Calculating %s (%s)" % (metrics.get_name(), self.neighborhood_mode))
            first_activity_dates, last_activity_dates = self._get_activity_dates()
            if metrics.value in [Metrics.NEIGHBORHOOD_QUALITY]:
                users_selection = self.select_users(NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS,
                                                    degree_in_static,
                                                    percent=0.1)
                data = metrics.calculate(self.authors.keys(), first_activity_dates, last_activity_dates,
                                         users_selection=users_selection)
            else:
                data = metrics.calculate(self.authors.keys(), first_activity_dates, last_activity_dates)
            log_fun("Calculated " + metrics.get_name() + ". Saving...")

            for i, user_id in enumerate(sorted(self.authors.keys())):  # For each author (node)
                print("Saving: ", i)
                d = data[user_id]
                m = modify_data(d, condition_fun)
                self._save_to_database(self.neighborhood_mode.short() + "_" + metrics.get_name(), user_id, m)
        except Exception as e:
            print('Exception saving:', e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        self._save_histograms_to_file(str(self.neighborhood_mode.value))
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

    def get_category(self, users_selection=None):
        # Get data
        data = self._databaseEngine.get_array_value_column("po_in_degree_static")
        users_selection = data.keys() if users_selection is None else users_selection
        # Categorize data
        bins = np.asarray([0, 10, 100, 1000])
        return {k: np.digitize([v], bins, right=True)[0] for k, v in data.items() if k in users_selection}

    def get_data(self, neighborhood_mode, metrics, fun=None,
                 cut_down=float("-inf"), cut_up=float("inf"), users_selection=None):
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
        data = self._databaseEngine.get_array_value_column(neighborhood_mode.short() + "_" + metrics.get_name(), fun)
        users_selection = data.keys() if users_selection is None else users_selection
        # Cut data
        if fun != fun_all:
            return {k: v for k, v in data.items() if cut_down < v < cut_up and k in users_selection}
        return {k: v for k, v in data.items() if k in users_selection}

    def correlation(self, neighborhood_mode, metrics, functions, do_abs=True, log_fun=logging.info):
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

                    df[m.get_name() + "_" + f.__name__] = [data[key] for key in keys]
            else:
                data = self._databaseEngine.get_array_value_column(neighborhood_mode.short() + "_" + m.get_name())
                df[m.get_name()] = [data[key] for key in keys]

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
                log_fun("\t" + str(r) + str(c_rate[key][r]))

        plt.show()

    def table(self, neighborhood_mode, metrics, functions, title='table', table_mode="index", log_fun=logging.info):
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
        log_fun('Table (' + table_mode + ') for ' + str([m.get_name() for m in metrics]) + " ...")
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)
        keys = sorted(self.authors.keys())
        df = pd.DataFrame()
        df['Name'] = [self.authors[key] for key in keys]
        try:
            for m, f in zip(metrics, functions):
                print(m.get_name() + f.__name__ if f is not None else '')
                if m.graph_iterator.graph_mode[0] == GraphIterator.ITERATOR.DYNAMIC \
                        or m.graph_iterator.graph_mode[0] == GraphIterator.ITERATOR.DYNAMIC_CURR_NEXT:
                    data = self._databaseEngine.get_array_value_column(neighborhood_mode.short() + "_" + m.get_name(),
                                                                       f)
                    df[m.get_name() + "_" + f.__name__] = [data[key] for key in keys]
                    if table_mode is not "value":
                        df[m.get_name() + "_" + f.__name__] = df[m.get_name() + "_" + f.__name__].rank(ascending=False)
                else:
                    data = self._databaseEngine.get_array_value_column(neighborhood_mode.short() + "_" + m.get_name())
                    df[m.get_name()] = [data[key] for key in keys]
                    if table_mode is not "value":
                        df[m.get_name()] = df[m.get_name()].rank(ascending=False)
            if table_mode is not "name":
                result = df
            else:
                metrics_keys = df.keys()
                authors_rankings = pd.DataFrame()
                for m in metrics_keys:
                    if m != 'Name':
                        authors_rankings[m] = list(df.sort_values(m, ascending=False)['Name'])
                result = authors_rankings

            if not os.path.exists('output'):
                os.mkdir('output')

            if not os.path.exists('output/table'):
                os.mkdir('output/table')

            result.to_csv(r'output/table/' + table_mode + "_" + title + '.txt', index=True, header=True)
            log_fun('Saved (saved in output/table)')
        except Exception as e:
            print(e)

    def display_between_range(self, neighborhood_mode, metrics, minimum, maximum, stats_fun=None, log_fun=logging.info):
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
            data = self._databaseEngine.get_array_value_column(neighborhood_mode.short() + "_" + metrics.get_name(),
                                                               stats_fun)
            for item in sorted(data.items(), key=lambda k: (k[1], k[0]), reverse=True):
                if minimum < item[1] < maximum:
                    log_fun('\t' + str(self.authors[item[0]]) + " " + str(item[1]))

    def select_users(self, neighborhood_mode, metrics, values_start=None, values_stop=None, percent=None,
                     selection=None):
        data = self._databaseEngine.get_array_value_column(neighborhood_mode.short() + "_" + metrics.get_name())
        sorted_keys = [item[0] for item in sorted(data.items(), key=lambda k: (k[1], k[0]))]
        sorted_keys.reverse()
        if selection is not None:
            sorted_keys = [k for k in sorted_keys if k in selection]
        if percent is None:
            return [k for k in sorted_keys if values_start <= data[k] <= values_stop]
        else:
            return sorted_keys[0:math.ceil(len(sorted_keys) * percent)]

    def _prepare_clustering_data(self, parameters, users_selection=None):
        """
        Prepares data about parameters names in a form of an array, which can be used in clustering.
        :param parameters: array (string)
            Parameters included in clustering.
        :return: numpy.array
            Data for clustering.
        """
        try:
            clustering_data = defaultdict(list)
            _clustering_data = {}
            users_selection = self.authors.keys() if not users_selection else users_selection
            dynamic_data = self._databaseEngine.get_array_value_column("po_in_degree_dynamic", len)
            intervals = {k: (dynamic_data[k] - 1  # One less because of curr_next compatibility
                             if dynamic_data[k] > 0 else 0)
                         for k in sorted(users_selection)}
            names = []
            for parameter in parameters:
                neighborhood_mode = parameter[0]
                metrics = parameter[1]
                weight = parameter[2]

                f = None if len(parameter) == 3 else parameter[3]
                data_name = neighborhood_mode.short() + "_" + metrics.get_name()
                data_name_f = data_name + ('' if f is None else "_" + f.__name__)
                names.append(data_name_f)

                data = self._databaseEngine.get_array_value_column(data_name, f)

                if metrics.get_name().endswith("dynamic") or metrics.get_name().endswith("dynamic_curr_next"):
                    for key in intervals.keys():
                        for i in range(min(len(data[key]), intervals[key])):
                            clustering_data[data_name_f].append(data[key][i])
                else:
                    clustering_data[data_name_f] = [data[k] for k in sorted(data.keys()) if k in users_selection]
                    intervals = None

                minimum, maximum = min(clustering_data[data_name_f]), max(clustering_data[data_name_f])
                _clustering_data[data_name_f] = clustering_data[data_name_f].copy()
                clustering_data[data_name_f] = [(x - minimum) / (maximum - minimum) * weight for x
                                                in clustering_data[data_name_f]]

            clustering_data = np.column_stack(clustering_data[p] for p in names)
            _clustering_data = np.column_stack(_clustering_data[p] for p in names)
            return clustering_data, _clustering_data, names, intervals
        except Exception as e:
            print("Prepare:", e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def random_forest_prediction(self, parameters, users_selection=None, log_fun=logging.info):
        try:
            pd.set_option('display.max_columns', None)

            def rmsle(ytrue, ypred):
                return np.sqrt(mean_squared_log_error(ytrue, ypred))

            users_selection = self.authors.keys() if not users_selection else users_selection
            users_selection = sorted(users_selection)
            # Degree will be predicted
            degree_in_dynamic = self._databaseEngine.get_array_value_column("po_in_degree_dynamic", fun_all)
            additional_data = {}
            for parameter in parameters:
                neighborhood_mode = parameter[0]
                metrics = parameter[1]
                data_name = neighborhood_mode.short() + "_" + metrics.get_name()
                d = self._databaseEngine.get_array_value_column(data_name, fun_all)  # Get all data (dynamic)
                additional_data[data_name] = d
                # d if data_name.endswith("dynamic") else {key: [np.nan] + d[key] for key in d}
            data_dict = defaultdict(list)
            for key in users_selection:
                # degree_in_dynamic[key] -> array of values for user
                # We iterate from back (all users have last interval)
                for i, value in enumerate(reversed(degree_in_dynamic[key])):
                    data_dict['Id'].append(key)
                    data_dict['Degree'].append(value)
                    data_dict['Interval'].append(len(GraphIterator.dynamic_graphs) - i)
                    for p in additional_data:
                        if 0 < i - 1 < len(additional_data[p][key]):
                            data_dict[p].append(additional_data[p][key][i - 1])
                        else:
                            data_dict[p].append(np.nan)
            data = pd.DataFrame.from_dict(data_dict)
            data = data.sort_values(by=['Id', 'Interval'])

            data2 = data.copy()
            data2['Last_Interval_Degree'] = data2.groupby(['Id'])['Degree'].shift()
            data2['Last_Interval_Diff'] = data2.groupby(['Id'])['Last_Interval_Degree'].diff()

            # for p in additional_data:
            #     data2[p + "_Diff"] = data2.groupby(['Id'])[p].diff()

            data2 = data2.dropna()
            print(data2.head(20))

            # Baseline score (prediction is equal last known value)
            print("Baseline")
            mean_error = []
            prediction_len = 50
            unknown_intervals = [i for i in range(max(data2['Interval']) - prediction_len, max(data2['Interval']) + 1)]
            train = data2[~data2['Interval'].isin(unknown_intervals)]
            print(data2['Interval'].max(), train['Interval'].max())
            # Last known interval
            prediction = train[train['Interval'] == train['Interval'].max()]
            for unknown_interval in unknown_intervals:
                test = data2[data2['Interval'] == unknown_interval]
                # Remove prediction for users who were not active
                test = test[test['Id'].isin(prediction['Id'])]
                prediction_test = prediction[prediction['Id'].isin(test['Id'])]
                print(len(test), len(prediction_test), len(prediction))
                error = rmsle(test['Degree'].values, prediction_test['Degree'].values)
                print('\tInterval %d - Error %.5f' % (unknown_interval, error))
                mean_error.append(error)
            print('\tMean Error = %.5f' % np.mean(mean_error))

            # X = df.drop('Survived', axis=1)
            # y = df['Survived']
            # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

            # Decision tree
            print("Decision tree")
            mean_error = []
            xtr = train.drop(['Degree'], axis=1)
            ytr = train['Degree'].values
            # Create model
            mdl = DecisionTreeClassifier()
            mdl.fit(xtr, ytr)
            for unknown_interval in unknown_intervals:
                test = data2[data2['Interval'] == unknown_interval]

                xts = test.drop(['Degree'], axis=1)
                yts = test['Degree'].values

                prediction = mdl.predict(xts)
                error = rmsle(yts, prediction)
                print('\tInterval %d - Error %.5f' % (unknown_interval, error))
                mean_error.append(error)
            print('\tMean Error = %.5f' % np.mean(mean_error))

        except Exception as e:
            print("Prediction:", e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

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
        try:
            log_fun('Clustering: k-means, n_clusters= %s...' % str(n_clusters) + '\n')
            data, _data, names, intervals = self._prepare_clustering_data(parameters, users_selection=users_selection)
            log_fun('\tData prepared.')
            k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
            k_means.fit(data)
            log_fun('\tData fitted.')
            if intervals is not None:
                self._save_dynamic_clusters(names, k_means.labels_, n_clusters, _data,
                                            intervals=intervals,
                                            users_selection=users_selection,
                                            log_fun=log_fun)
            else:
                self._save_clusters(names, k_means.labels_, n_clusters, _data, users_selection, log_fun=log_fun)
            log_fun('Clustering finished. Result saved in output folder.')
        except Exception as e:
            print("Save:", e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def agglomerative_clustering(self, n_clusters, parameters, users_selection=None, log_fun=logging.info):
        try:
            log_fun('Clustering: agglomerative clustering n_clusters= %s...' % str(n_clusters) + '\n')
            data, _data, names, intervals = self._prepare_clustering_data(parameters, users_selection)
            log_fun('\tData prepared.')
            clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(data)
            # clustering = AgglomerativeClustering(compute_full_tree=True, distance_threshold=1.0,
            # n_clusters=None).fit(data)
            log_fun('\tData fitted.')
            if intervals is not None:
                self._save_dynamic_clusters(names, clustering.labels_, n_clusters, _data,
                                            intervals=intervals,
                                            users_selection=users_selection,
                                            log_fun=log_fun)
            else:
                self._save_clusters(names, clustering.labels_, n_clusters, _data, users_selection, log_fun=log_fun)

            log_fun('Clustering finished. Result saved in output folder.')
        except Exception as e:
            print("Save:", e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

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

    def select_post_count(self, day_start, day_end):
        self.posts_counts.append(self._databaseEngine.get_posts(day_start, day_end))

    def select_responses_count(self, day_start, day_end):
        self.responses_counts.append(self._databaseEngine.get_responses(day_start, day_end))

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
        for day in (self._dates_range[0] + dt.timedelta(n) for n in range(self._days_count)):
            day_start = day.replace(hour=00, minute=00, second=00)
            day_end = day.replace(hour=23, minute=59, second=59)
            days.append(day_start.date())
            self._select_comments(day_start, day_end)
            self.select_post_count(day_start, day_end)
            self.select_responses_count(day_start, day_end)
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
            graph.add_attribute('posts', sum_by_key(self.posts_counts))
            graph.add_attribute('responses', sum_by_key(self.responses_counts))
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
                graph.add_attribute('posts', sum_by_key(self.posts_counts[i:end + 1]))
                graph.add_attribute('responses', sum_by_key(self.responses_counts[i:end + 1]))
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

    class Cluster:
        def __init__(self):
            pass

    def _create_and_save_series(self, keys, intervals, classes, filename):
        try:
            # Create dataframe with time series
            series_dict = defaultdict(list)
            users_ids = []
            names = []

            # Prepare id label for each class item
            for key in keys:
                users_ids.extend([key] * intervals[key])
                names.append(self.authors[key])

            # Transform classes array into series
            for u_id, c in zip(users_ids, classes):
                series_dict[u_id].append(c)

            # Append -1 at the beginning (otherwise there would be nan at the end)
            for u_id in users_ids:
                u_len = len(GraphIterator.dynamic_graphs) - len(series_dict[u_id])
                series_dict[u_id] = [-1] * u_len + series_dict[u_id]

            # Create DataFrame from dict
            series = pd.DataFrame.from_dict(series_dict, orient='index')
            series.insert(0, "Name", names)

            # Save to file
            if not os.path.exists('output/clustering_dynamic'):
                os.mkdir('output/clustering_dynamic')
            series.to_csv(r'output/clustering_dynamic/' + filename, index=True, header=True)

            return series_dict, series, users_ids
        except Exception as e:
            print("Save:", e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def _check_interesting_users(self, series, n_clusters):
        try:
            selected_in_cluster = defaultdict(list)

            interesting_users_50 = self._databaseEngine.get_interesting_users(50)
            interesting_users_250 = self._databaseEngine.get_interesting_users(250)
            interesting_users_500 = self._databaseEngine.get_interesting_users(500)
            interesting_users_1000 = self._databaseEngine.get_interesting_users(1000)

            # Cluster: Occurrences
            len_50 = defaultdict(int)
            len_250 = defaultdict(int)
            len_500 = defaultdict(int)
            len_1000 = defaultdict(int)

            len_50_s = defaultdict(int)
            len_250_s = defaultdict(int)
            len_500_s = defaultdict(int)
            len_1000_s = defaultdict(int)

            # Check interesting users
            for user in interesting_users_1000:
                # How many times interesting was in each cluster
                c = Counter(list(series.loc[user]))
                for cluster in range(n_clusters):
                    if cluster in c:
                        if user in interesting_users_50:
                            len_50[cluster] += c[cluster]
                            len_50_s[cluster] += 1
                            selected_in_cluster[cluster].append(user)
                        if user in interesting_users_250:
                            len_250[cluster] += c[cluster]
                            len_250_s[cluster] += 1
                        if user in interesting_users_500:
                            len_500[cluster] += c[cluster]
                            len_500_s[cluster] += 1
                        if user in interesting_users_1000:
                            len_1000[cluster] += c[cluster]
                            len_1000_s[cluster] += 1

            return selected_in_cluster, len_50, len_50_s, len_250, len_250_s, len_500, len_500_s, len_1000, len_1000_s
        except Exception as e:
            print("Save:", e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    @staticmethod
    def _analyse_cluster_sizes(series, n_clusters):
        try:
            # Check cluster sizes
            sizes_dict = defaultdict(list)
            for (columnName, columnData) in series.iteritems():
                for c in range(n_clusters):
                    sizes_dict[c].append(list(columnData.values).count(c))

            # Plot cluster sizes changes
            plt.figure()
            for c in range(n_clusters):
                plt.plot(sizes_dict[c], label=c + 1)
            plt.legend()
            plt.show()

            # Plot percentage of cluster sizes changes
            counts = pd.DataFrame.from_dict(sizes_dict)
            data_perc = counts.divide(counts.sum(axis=1), axis=0)

            plot_data = []
            plot_labels = []
            plt.figure()
            for (columnName, columnData) in data_perc.iteritems():
                plot_data.append(list(columnData.values))
                plot_labels.append(columnName)
            plt.stackplot(range(len(counts[0])), *plot_data, labels=plot_labels)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.margins(0, 0)
            plt.show()
        except Exception as e:
            print("Save:", e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    @staticmethod
    def _analyse_transitions(keys, series_dict, timestr):
        try:
            transitions = defaultdict(list)
            t_keys = set()

            # Create dataframe of transmisions between clusters
            for key in keys:
                for i in range(len(series_dict[key]) - 1):
                    c_1 = series_dict[key][i]
                    c_2 = series_dict[key][i + 1]
                    if c_1 != -1 and c_2 != -1:
                        k = str(c_1 + 1) + " " + str(c_2 + 1)
                        transitions[key].append(k)
                        t_keys.add(k)
                    else:
                        transitions[key].append("-1 -1")
            trans = pd.DataFrame.from_dict(transitions, orient='index')
            # df.insert(0, "Name", names)
            if not os.path.exists('output/clustering_dynamic'):
                os.mkdir('output/clustering_dynamic')
            trans.to_csv(r'output/clustering_dynamic/' + timestr + 'user_transitions' + '.txt', index=True, header=True)
            print("transitions saved")

            # Check transitions sizes
            counts_t = defaultdict(list)
            for (columnName, columnData) in trans.iteritems():
                c = Counter(list(columnData.values))
                for key in sorted(t_keys):
                    counts_t[key].append(c[key] if key in c else 0)
            print("transitions counts calculated")
            trans_c = pd.DataFrame.from_dict(counts_t, orient='index')
            if not os.path.exists('output/clustering_dynamic'):
                os.mkdir('output/clustering_dynamic')
            trans_c.to_csv(r'output/clustering_dynamic/' + timestr + 'user_transitions_counts' + '.txt', index=True,
                           header=True)
            print("transitions counts saved")

            # Plot cluster sizes changes
            plt.figure()
            colors = [(252, 3, 3), (252, 74, 3), (252, 152, 3), (252, 231, 3), (202, 252, 3),
                      (115, 252, 3), (3, 252, 20), (3, 252, 111), (3, 252, 202), (3, 211, 252),
                      (3, 111, 252), (3, 24, 252), (103, 3, 252), (173, 3, 252), (252, 3, 235),
                      (252, 3, 132), (252, 3, 69), (133, 0, 0), (133, 51, 0), (133, 98, 0),
                      (129, 133, 0), (86, 133, 0), (0, 92, 9), (0, 92, 66), (0, 75, 92), (0, 23, 92),
                      (40, 0, 54), (54, 0, 32), (54, 0, 9), (110, 78, 70), (110, 107, 70),
                      (76, 110, 70), (70, 110, 110), (70, 81, 110), (107, 70, 110), (86, 87, 85),
                      (145, 145, 145), (186, 186, 186), (187, 245, 184), (184, 245, 229), (106, 166, 150)
                      ]

            colors = [(e[0] / 255, e[1] / 255, e[2] / 255) for e in colors]
            for i, key in enumerate(sorted(t_keys)):
                plt.plot(counts_t[key], label=key, color=colors[i % len(colors)])
            plt.legend(ncol=2)

            plt.show()

            # Plot percentage of cluster transitions
            counts = pd.DataFrame.from_dict(counts_t)
            data_perc = counts.divide(counts.sum(axis=1), axis=0)

            plot_data = []
            plot_labels = []
            plt.figure()
            for (columnName, columnData) in data_perc.iteritems():
                plot_data.append(list(columnData.values))
                plot_labels.append(columnName)
            plt.stackplot(range(len(counts[list(t_keys)[0]])), *plot_data, labels=plot_labels, colors=colors)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
            plt.margins(0, 0)
            plt.show()
        except Exception as e:
            print("Save:", e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def _save_dynamic_clusters(self, parameters_names, classes, n_clusters, data, intervals,
                               users_selection=None, log_fun=logging.debug):
        """
        Displays results of clustering.
        :param parameters_names: array (string)
            Parameters included in clustering.
        :param classes: array
            Classes created
        :param n_clusters: int
            Number of clusters
        """
        timestr = time.strftime("%Y%m%d-%H%M%S")
        keys = sorted(self.authors.keys()) if not users_selection else np.array(sorted(users_selection))

        try:
            series_dict, series, users_ids = self._create_and_save_series(keys, intervals, classes,
                                                                          timestr + 'user_clusters' + '.txt')

            selected_in_cluster, len_50, len_50_s, len_250, len_250_s, len_500, len_500_s, len_1000, len_1000_s \
                = self._check_interesting_users(series, n_clusters)

            self._analyse_cluster_sizes(series, n_clusters)
            self._analyse_transitions(keys, series_dict, timestr)

            ranges = defaultdict(list)
            p_values = defaultdict(list)
            save = defaultdict(list)

            for cluster in range(n_clusters):
                print(cluster)

                indexes = np.where(classes == cluster)
                users_in_cluster = np.array(users_ids)[indexes]

                log_fun('Cluster: %s' % (cluster + 1))
                log_fun('\t Number of users: %s %s' % (len(users_in_cluster), len(set(users_in_cluster))))
                log_fun('   Parameters:')

                save['stats'].extend(['min', 'max', 'mean', 'stdev'])
                empty_stats = ['' for _ in range(3)]
                save['cluster'].append(cluster)
                save['cluster'].extend(empty_stats)
                save['size'].append(len(users_in_cluster))
                save['size'].append(len(set(users_in_cluster)))
                save['size'].extend(empty_stats[:-1])

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

                log_fun("   Sample users:")

                log_fun("\t 50= %s, 250= %s, 500= %s, 1000= %s" % (len_50[cluster], len_250[cluster],
                                                                   len_500[cluster], len_1000[cluster]))
                log_fun('\n')
                save['sample'].extend([selected_in_cluster[cluster], *empty_stats])
                save['first 50'].extend([len_50[cluster], len_50_s[cluster], *empty_stats[:-1]])
                save['first 250'].extend([len_250[cluster], len_250_s[cluster], *empty_stats[:-1]])
                save['first 500'].extend([len_500[cluster], len_500_s[cluster], *empty_stats[:-1]])
                save['first 1000'].extend([len_1000[cluster], len_1000_s[cluster], *empty_stats[:-1]])

            file_name = timestr + str(n_clusters) + ".txt"
            FileWriter.write_split_row_to_file(FileWriter.CLUSTERING, file_name, ['cluster', save['cluster']])
            FileWriter.write_split_row_to_file(FileWriter.CLUSTERING, file_name, ['size', save['size']])
            FileWriter.write_split_row_to_file(FileWriter.CLUSTERING, file_name, ['stats', save['stats']])
            for parameter_name in parameters_names:
                FileWriter.write_split_row_to_file(FileWriter.CLUSTERING, file_name, [parameter_name,
                                                                                      save[parameter_name]])
            FileWriter.write_split_row_to_file(FileWriter.CLUSTERING, file_name, ['sample', save['sample']])
            FileWriter.write_split_row_to_file(FileWriter.CLUSTERING, file_name, ['first 50', save['first 50']])
            FileWriter.write_split_row_to_file(FileWriter.CLUSTERING, file_name, ['first 250', save['first 250']])
            FileWriter.write_split_row_to_file(FileWriter.CLUSTERING, file_name, ['first 500', save['first 500']])
            FileWriter.write_split_row_to_file(FileWriter.CLUSTERING, file_name, ['first 1000', save['first 1000']])

            Manager.plot_overlapping(parameters_names, p_values, True)
            plt.show()
        except Exception as e:
            print("Save:", e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

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
            print(cluster)
            try:
                indexes = np.where(classes == cluster)
                users_ids = np.array(sorted(self.authors.keys()))[indexes] if not users_selection \
                    else np.array(sorted(users_selection))[indexes]

                log_fun('Cluster: %s' % (cluster + 1))
                log_fun('\t Number of users: %s' % (len(users_ids)))
                log_fun('   Parameters:')

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

                log_fun("   Sample users:")
                s = []
                len_50 = 0
                len_250 = 0
                len_500 = 0
                len_1000 = 0

                for i in users_ids:
                    if i in interesting_users_50:
                        s.append(self.authors[i])
                        # parameters = [
                        #     self._databaseEngine.get_array_value_column_for_user(parameter_name, i, mean)
                        #     for parameter_name in parameters_names]
                        log_fun("\t %s" % self.authors[i])  # [round(p, 3) for p in parameters]))
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
                log_fun("\t 50= %s, 250= %s, 500= %s, 1000= %s" % (len_50, len_250, len_500, len_1000))
                log_fun('\n')
                save['sample'].extend([s, *empty_stats])
                save['first 50'].extend([len_50, len_50 / len(users_ids) * 100, *empty_stats[:-1]])
                save['first 250'].extend([len_250, len_250 / len(users_ids) * 100, *empty_stats[:-1]])
                save['first 500'].extend([len_500, len_500 / len(users_ids) * 100, *empty_stats[:-1]])
                save['first 1000'].extend([len_1000, len_1000 / len(users_ids) * 100, *empty_stats[:-1]])
            except Exception as e:
                print(e)
        try:
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
            FileWriter.write_split_row_to_file(FileWriter.CLUSTERING, str(n_clusters),
                                               ['first 1000', save['first 1000']])
        except Exception as e:
            print("Save:", e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

        # Manager.plot_overlapping(parameters_names, ranges, False)
        try:
            Manager.plot_overlapping(parameters_names, p_values, True)
            plt.show()
        except Exception as e:
            print(e)

    @staticmethod
    def plot_overlapping(parameters_names, arr, points=False):
        try:
            if len(parameters_names) > 3:
                x, y = math.ceil(len(parameters_names) / 2), 2
                fig, axs = plt.subplots(x, y)
            else:
                fig, axs = plt.subplots(len(parameters_names))
            plots = None
            for i, key in enumerate(parameters_names):
                a = axs[i] if len(parameters_names) <= 3 else axs[int(i / 2), i % 2]
                plots = \
                    Manager.plot_overlapping_ranges_by_points(arr[key], a) if points \
                        else Manager.plot_overlapping_ranges(arr[key], a)

                a.set_title(key, fontsize=6)
            fig.legend(plots, labels=['Cluster ' + str(i + 1) for i in range(len(arr[parameters_names[-1]]))],
                       loc="center right",
                       bbox_to_anchor=(0.99, 0.5), borderaxespad=0.1, prop={'size': 6})
            fig.tight_layout()
            plt.subplots_adjust(right=0.85)

            if len(parameters_names) % 2 != 0:
                axs[-1, -1].axis('off')
        except Exception as e:
            print(e)

    @staticmethod
    def plot_overlapping_ranges_by_points(points, axs):
        axs.set_ylim([-2, 2 * len(points) + 2])
        axs.yaxis.set_visible(False)

        for item in axs.get_xticklabels():
            item.set_fontsize(6)

        x = points
        y = [(len(points) - i) * 2 for i in range(len(points))]

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

    def _get_activity_dates(self):
        """
        Gets first_activity_date column.
        :return: dict
        """
        return self._databaseEngine.get_activity_dates()
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
