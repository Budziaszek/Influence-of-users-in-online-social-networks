import datetime as dt
import os
import pickle
import warnings
from pathlib import Path
import numpy
from statsmodels.tsa.statespace.tools import diff
from sklearn.cluster import KMeans
from statistics import mean, stdev

from Metrics.MetricsProcessing.Histogram import Histogram
from Metrics.MetricsType import MetricsType
from Network.GraphConnectionType import GraphConnectionType
from Metrics.MetricsProcessing.Prediction import Prediction
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
    mode = None
    authors_ids = None
    authors_names = None
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
            # print("Creating histogram for ", fun_name)
            self.file_name = fun_name + "_" + "hist_" + calculated_value.get_name() + ".txt"
            self._histogram = Histogram(x_scale, size_scale)

        def add_data(self, size, data):
            self._histogram.add(size, data, self.data_function)

        def save(self, folder):
            self._histogram.save(folder, self.file_name)

    def __init__(self, parameters, test=False):
        """
        Creates DatabaseEngine that connects to database with given parameters and later allow operations
        (e.g. queries). Define time intervals. Initializes some variables for future use.
        :param parameters: str
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
        self._databaseEngine.connect(parameters)
        self._databaseEngine.create_first_activity_date_column()
        self._dates_range = self._get_dates_range()
        self._days_count = (self._dates_range[1] - self._dates_range[0]).days
        if test is True:
            self._days_count = self._number_of_days_in_interval * 5
        self.authors_ids = self._get_authors("id")
        self.authors_names = self._get_authors("name")
        self.static_neighborhood_size = None

    def calculate(self, mode, metrics,
                  save_to_file=True, save_to_database=True, predict=False, calculate_histogram=False,
                  x_scale=None, size_scale=None, data_condition_function=None, data_functions=None):
        """
        Calculates metrics values for each user and allows creating files and saving to database.
        :param mode: NeighborhoodMode
            Defines model mode (which comments should me included)
        :param save_to_database: bool
            True - save calculated value to database
        :param metrics: MetricsType
            Calculate function from given class is called
        :param save_to_file: bool
            True - save full data to file
        :param predict: bool
            True - predict time series
        :param calculate_histogram: bool
            True - calculate and save data as a histogram
        :param x_scale: array (float)
            Defines standard classes for the histogram
        :param size_scale: array (int)
            Defines size classes for the histogram
        :param data_condition_function: function
            Condition function defines which values should be removed to e.g. remove None values
        :param data_functions: array (function)
            DataProcessing function defines how data should be modified in order to aggregate them e.g. minimum
        """
        if mode != self.mode:
            self._create_graphs(mode)

        file_writer = self._initialize_file_writer(save_to_file, metrics)
        self._initialize_histogram_managers(calculate_histogram, data_functions, metrics, x_scale, size_scale)
        bar = ProgressBar("Calculating %s (%s)" % (metrics.get_name(), self.mode),
                          "Calculated", len(self.authors_ids))
        for i in range(len(self.authors_ids)):  # For each author (node)
            bar.next()
            first_activity_date = self._get_first_activity_date(self.authors_ids[i])
            data = metrics.calculate(self.authors_ids[i], first_activity_date)
            if predict:
                self.predict(data, i)
            data_modified = modify_data(data, data_condition_function) \
                if calculate_histogram or save_to_file or save_to_database else []
            if calculate_histogram:
                self._add_data_to_histograms(self.static_neighborhood_size[i], data_modified)
            self._save_data_to_file(file_writer, i, data_modified)
            if save_to_database:
                self._save_to_database(metrics.get_name(), self.authors_ids[i], data_modified)
        self._save_histograms_to_file(str(self.mode.value))
        bar.finish()

    def process_loaded_data(self, metrics, predict=False, calculate_histogram=False,
                            x_scale=None, size_scale=None, data_condition_function=None, data_functions=None):
        """
        Load data from database - only if metrics was calculated before. Result can be used in prediction or histogram.
        :param metrics: MetricsType
            Calculate function from given class is called
        :param predict: bool
            True - predict time series
        :param calculate_histogram: bool
            True - calculate and save data as a histogram
        :param x_scale: array (float)
            Defines standard classes for the histogram
        :param size_scale: array (int)
            Defines size classes for the histogram
        :param data_condition_function: function
            Condition function defines which values should be removed to e.g. remove None values
        :param data_functions: array (function)
            DataProcessing function defines how data should be modified in order to aggregate them e.g. minimum
        """
        self._initialize_histogram_managers(calculate_histogram, data_functions, metrics, x_scale, size_scale)
        bar = ProgressBar("Processing %s" % metrics.value, "Calculated", len(self.authors_ids))
        for i in range(len(self.authors_ids)):  # For each author (node)
            bar.next()
            data = self._databaseEngine.get_array_value_column_for_user(metrics.get_name(), self.authors_ids[i], None)
            if predict:
                self.predict(data, i)
            data_modified = modify_data(data, data_condition_function) if calculate_histogram else []
            if calculate_histogram:
                self._add_data_to_histograms(self.static_neighborhood_size[i], data_modified)
        for histogram in self.histogram_managers:
            histogram.save('output/' + str(self.mode.value) + "/")
        bar.finish()

    def k_means(self, n_clusters, parameters_names):
        """
        Performs k-means clustering and displays results.
        :param n_clusters: int
            Number of clusters.
        :param parameters_names: array (string)
            Parameters included in clustering.
        """
        print(Manager.background + 'Clustering: k-means (n_clusters= %s)' % str(n_clusters) + '\n' + Manager.reset)
        data = self._prepare_clustering_data(parameters_names)
        k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
        k_means.fit(data)
        self._display_clustering_results(parameters_names, k_means.predict(data), k_means.cluster_centers_, data)

    def predict(self, data, author_i):
        """
        Predict time series.
        :param data: array
            DataProcessing used in prediction
        :param author_i: int
            Index of author
        """
        plot_data = []
        title_data = []
        interesting_ids = [1672, 440, 241, 2177, 797, 3621, 11, 2516]
        # methods = [Prediction.exponential_smoothing, Prediction.ARIMA]
        parameters_versions = [i for i in range(3)]
        data = make_data_positive(diff(data))
        # print("Predict")
        prediction = Prediction(data, self.authors_names[author_i])
        for parameters_version in parameters_versions:
            result = prediction.predict(0, len(data) - 50, 50, Prediction.exponential_smoothing,
                                        Prediction.MAPE_error, parameters_version)
            if len(plot_data) == 0:
                plot_data.append((result[2].index, result[2]))
                title_data.append("Original")
            plot_data.append(result[1])
            title_data.append(result[0] + str(parameters_version))
        if self.authors_ids[author_i] in interesting_ids:
            Prediction.plot("Prediction for " + self.authors_names[author_i], plot_data, title_data)

    def _get_dates_range(self):
        """
        Checks dates range (find min, max action (comment or post) time and number of days)
        :return: tuple
            Contains first and last date occurring in database
        """
        date = "date"
        r = []
        for column in ["comments", "posts"]:
            r.append(self._databaseEngine.execute("SELECT min(" + date + "), max(" + date + ") FROM " + column).pop())
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
        c = self._databaseEngine.execute("""SELECT c.author_id, p.author_id 
                                            FROM comments c
                                            INNER JOIN posts p 
                                            ON c.post_id = p.id 
                                            WHERE c.date BETWEEN """
                                         + "'" + str(day_start) + "' and '" + str(day_end) + "'"
                                         + ("" if include_responses_from_author
                                            else "AND c.author_id!=p.author_id"))
        return numpy.array(c)

    def _select_responses_to_comments(self, day_start, day_end, include_responses_from_author):
        """
        Executes query, which selects comments added to comments.
        :param day_start: datetime.datetime
        :param day_end: datetime.datetime
        :param include_responses_from_author: bool
            True - reactions from author should be included
        """
        c = self._databaseEngine.execute("""SELECT c.author_id, p.author_id
                                            FROM comments c
                                            INNER JOIN comments p 
                                            ON c.parentcomment_id = p.id
                                            WHERE c.parentcomment_id IS NOT NULL
                                            AND c.date BETWEEN """
                                         + "'" + str(day_start) + "' and '" + str(day_end) + "'"
                                         + ("" if include_responses_from_author
                                            else "AND c.author_id!=p.author_id"))
        return numpy.array(c)

    def _select_comments(self, day_start, day_end):
        """
        Calls methods, which selects all required comments if they weren't already selected.
        :param day_start: datetime.datetime
        :param day_end: datetime.datetime
        """
        if self.mode.do_read_comments_to_posts:
            self._comments_to_posts.append_data(day_start, day_end)
        if self.mode.do_read_comments_to_comments:
            self._comments_to_comments.append_data(day_start, day_end)
        if self.mode.do_read_comments_to_posts_from_others:
            self._comments_to_posts_from_others.append_data(day_start, day_end)
        if self.mode.do_read_comments_to_comments_from_others:
            self._comments_to_comments_from_others.append_data(day_start, day_end)

    def _set_comments_to_add(self):
        """
        Sets variable, which contains comments included in model.
        """
        self.comments_to_add = []
        if self.mode is NeighborhoodMode.COMMENTS_TO_POSTS:
            self.comments_to_add.append(self._comments_to_posts.get_data())
        if self.mode is NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS:
            self.comments_to_add.append(self._comments_to_posts_from_others.get_data())
        if self.mode is NeighborhoodMode.COMMENTS_TO_COMMENTS:
            self.comments_to_add.append(self._comments_to_comments.get_data())
        if self.mode is NeighborhoodMode.COMMENTS_TO_COMMENTS_FROM_OTHERS:
            self.comments_to_add.append(self._comments_to_comments_from_others.get_data())
        if self.mode is NeighborhoodMode.COMMENTS_TO_POSTS_AND_COMMENTS:
            self.comments_to_add.append(self._comments_to_posts.get_data())
            self.comments_to_add.append(self._comments_to_comments.get_data())
        if self.mode is NeighborhoodMode.COMMENTS_TO_POSTS_AND_COMMENTS_FROM_OTHERS:
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
            print("Creating static graph and dynamic graphs")
            self._add_data_to_static_graph()
            self._add_data_to_dynamic_graphs()
        elif graph_type is "d":
            print("Creating dynamic graphs")
            self._add_data_to_dynamic_graphs()
        elif graph_type is "s":
            print("Creating static graph")
            self._add_data_to_static_graph()
        else:
            raise Exception("ERROR - wrong graph value")
        print("Graphs created")

    def _add_data_to_static_graph(self):
        """
        Adds data to static graph - all days.
        """
        graph = SocialNetworkGraph()
        for comments in self.comments_to_add:
            edges = numpy.concatenate(comments, axis=0)
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
                edges = numpy.concatenate(comments[i:end + 1], axis=0)
                graph.add_edges(edges)
            graph.start_day, graph.end_day = self.days[i], self.days[end]
            self.dynamic_graphs.append(graph)
            graph = SocialNetworkGraph()
            i += step

    def _create_graphs(self, mode):
        """
        Creates graphs corresponding to the selected mode.
        :param mode: NeighborhoodMode
            Defines model mode (which comments should me included)
        """
        self.mode = mode
        graphs_file_name = 'graphs' + "/" + self.mode.name \
                           + "_" + str(self._number_of_days_in_interval) \
                           + "_" + str(self._number_of_new_days_in_interval)
        Path('graphs').mkdir(parents=True, exist_ok=True)

        if os.path.exists(graphs_file_name):
            self._load_graphs_from_file(graphs_file_name)
            GraphIterator.set_graphs(self.static_graph, self.dynamic_graphs)
        else:
            self._read_salon24_comments_data_by_day()
            self._add_data_to_graphs("sd", False)
            GraphIterator.set_graphs(self.static_graph, self.dynamic_graphs)
            self._save_graphs_to_file(graphs_file_name)

    def _load_graphs_from_file(self, graphs_file_name):
        """
        Loads graphs from file.
        :param graphs_file_name:
            Filename from which graphs will be loaded
        """
        print("Loading graphs from file")
        with open(graphs_file_name, 'rb') as file:
            dictionary = pickle.load(file)
            self.static_graph = dictionary['static']
            self.dynamic_graphs = dictionary['dynamic']

    def _prepare_clustering_data(self, parameters_names):
        """
        Prepares data about parameters names in a form of an array, which can be used in clustering.
        :param parameters_names: array (string)
            Parameters included in clustering.
        :return: numpy.array
            Data for clustering.
        """
        data = {}
        for parameter_name in parameters_names:
            data[parameter_name] = numpy.array(self._databaseEngine.get_array_value_column(parameter_name, mean))
            minimum, maximum = min(data[parameter_name]), max(data[parameter_name])
            data[parameter_name] = [(x - minimum) / (maximum - minimum) for x in data[parameter_name]]
        return numpy.column_stack(data[parameter_name] for parameter_name in parameters_names)

    def _display_clustering_results(self, parameters_names, classes, centers, data):
        """
        Displays results of clustering.
        :param parameters_names: array (string)
            Parameters included in clustering.
        :param classes: array
            Classes created
        :param centers: array
            Centers of clusters
        """
        for cluster in range(len(centers)):
            indexes = numpy.where(classes == cluster)
            names = numpy.array(self.authors_names)[indexes]
            users_ids = numpy.array(self.authors_ids)[indexes]
            print('Cluster: %s' % cluster)
            print('\t center: %s\n\t number of users: %s' % ([round(c, 3) for c in centers[cluster]], len(names)))
            print('Parameters:')
            for i, parameters_name in enumerate(parameters_names):
                values = data[indexes, i][0]
                print("\t %s: min= %s, max= %s, mean= %s, stdev= %s" %
                      (parameters_name, round(min(values), 3), round(max(values), 3),
                       round(mean(values), 3), round(stdev(values), 3)))
            print("Sample users:")
            for i in range(min(len(names), 10)):
                parameters = [self._databaseEngine.get_array_value_column_for_user(parameter_name, users_ids[i], mean)
                              for parameter_name in parameters_names]
                print("\t %s: %s" % (names[i], [round(p, 3) for p in parameters]))
            print()

    def _save_data_to_file(self, file_writer, author_index, data):
        """
        Saves data to file.
        :param file_writer: FileWriter
            FileWriter used to save data
        :param author_index: int
            Index of the author in names and ids arrays
        :param data: array (float)
            Author's data
        """
        if file_writer is not None:
            file_writer.write_row_to_file([self.authors_ids[author_index], self.authors_names[author_index], *data])

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
            self._calculate_and_get_authors_static_neighborhood_size()
            for data_function in data_functions:
                self.histogram_managers.append(self.HistogramManager(data_function, value, x_scale, size_scale))

    def _initialize_file_writer(self, save_to_file, calculated_value):
        """
        Initializes FileWriter if required.
        :param save_to_file: bool
            True - initialization is required
        :param calculated_value: MetricsType
            Defines filename
        :return: FileWriter
            Initialized FileWriter if initialization is required otherwise None
        """
        if save_to_file:
            file_name = calculated_value.get_name() + ".txt"
            file_writer = FileWriter()
            file_writer.set_all(self.mode.value, file_name, self._get_graphs_labels(3))
            return file_writer
        return None

    def _get_authors(self, parameter):
        """
        Created array of authors parameters ordered by id.
        :param parameter: str
            Parameter which should be selected from database
        :return: array
            Array with authors parameters ordered by id
        """
        return [x[0] for x in self._databaseEngine.execute("SELECT " + parameter + " FROM authors ORDER BY id")]

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

    def _get_first_activity_date(self, author_id):
        """
        Checks author's first activity date and returns it.
        :param author_id:
            Id of author
        :return: datetime.datetime
            First activity date
        """
        try:
            return self._databaseEngine.execute("SELECT %s FROM authors WHERE id = %s"
                                                % ("first_activity_date", author_id))[0][0]
        except IndexError:
            return None

    def _calculate_and_get_authors_static_neighborhood_size(self):
        """
        Calculates static neighborhood size and returns it.
        :return: array (int)
            Neighborhoods sizes
        """
        if self.static_neighborhood_size is None:
            calculated_value = MetricsType(MetricsType.NEIGHBORS_COUNT, GraphConnectionType.IN,
                                           GraphIterator(GraphIterator.GraphMode.STATIC))
            self.static_neighborhood_size = \
                {i: calculated_value.calculate(self.authors_ids[i],
                                               self._get_first_activity_date(self.authors_ids[i]))[0]
                 for i in range(len(self.authors_ids))}
        return self.static_neighborhood_size

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
            Mode name - folder
        """
        for histogram in self.histogram_managers:
            histogram.save('output/' + mode_name + "/")
