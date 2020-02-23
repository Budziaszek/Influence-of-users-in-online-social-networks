import datetime as dt
import os
import pickle
from pathlib import Path

from Histogram import Histogram
from MetricsType import GraphIterator, MetricsType
from Network.GraphConnectionType import GraphConnectionType
from Prediction import Prediction
from ProgressBar import ProgressBar
from Data.PostgresDatabaseEngine import PostgresDatabaseEngine
from Network.SocialNetworkGraph import SocialNetworkGraph
from Data.FileWriter import FileWriter
from Mode import Mode
import numpy
from statsmodels.tsa.statespace.tools import diff


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
        self._comments_to_comments = self.CommentsReader(self._select_responses_to_comments, True)
        self._comments_to_posts = self.CommentsReader(self._select_responses_to_posts, True)
        self._comments_to_comments_from_others = self.CommentsReader(self._select_responses_to_comments, False)
        self._comments_to_posts_from_others = self.CommentsReader(self._select_responses_to_posts, False)

        self._databaseEngine.connect(parameters)
        self._databaseEngine.create_first_activity_date_column()
        self._dates_range = self._get_dates_range()
        self._days_count = (self._dates_range[1] - self._dates_range[0]).days
        if test is True:
            self._days_count = self._number_of_days_in_interval * 5
        self.authors_ids = self.get_authors("id")
        self.authors_names = self.get_authors("name")
        self.authors_static_neighborhood_size = None

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
        if self.mode is Mode.COMMENTS_TO_POSTS:
            self.comments_to_add.append(self._comments_to_posts.get_data())
        if self.mode is Mode.COMMENTS_TO_POSTS_FROM_OTHERS:
            self.comments_to_add.append(self._comments_to_posts_from_others.get_data())
        if self.mode is Mode.COMMENTS_TO_COMMENTS:
            self.comments_to_add.append(self._comments_to_comments.get_data())
        if self.mode is Mode.COMMENTS_TO_COMMENTS_FROM_OTHERS:
            self.comments_to_add.append(self._comments_to_comments_from_others.get_data())
        if self.mode is Mode.COMMENTS_TO_POSTS_AND_COMMENTS:
            self.comments_to_add.append(self._comments_to_posts.get_data())
            self.comments_to_add.append(self._comments_to_comments.get_data())
        if self.mode is Mode.COMMENTS_TO_POSTS_AND_COMMENTS_FROM_OTHERS:
            self.comments_to_add.append(self._comments_to_posts_from_others.get_data())
            self.comments_to_add.append(self._comments_to_comments_from_others.get_data())

    def _read_salon24_comments_data_by_day(self):
        """
        Retrieves the most important values about comments (tuple: (comment author, post author)) by day from database
        and store values in array (chronologically day by day)
        """
        days = []
        bar = ProgressBar("Selecting _data", "Data selected", self._days_count)
        for day in (self._dates_range[0] + dt.timedelta(n) for n in range(self._days_count)):
            day_start = day.replace(hour=00, minute=00, second=00)
            day_end = day.replace(hour=23, minute=59, second=59)
            days.append(day_start.date())
            self._select_comments(day_start, day_end)
            bar.next()
        bar.finish()
        self._set_comments_to_add()
        self.days = days

    def _add_data_to_graphs(self, graph_type, is_multi):
        """
        Adds selected comments data to graphs.
        :param graph_type: string
            Defines whenever static graph or dynamics graphs should be created: "s" - static, "d" - dynamic, "sd" - both
        :param is_multi: bool
            True - multi-graph
            False - single-edges graph
        """
        self.dynamic_graphs = []
        self.static_graph = None
        if graph_type is "sd":
            print("Creating static graph and dynamic graphs")
            self._add_data_to_static_graph(is_multi)
            self._add_data_to_dynamic_graphs(is_multi)
        elif graph_type is "d":
            print("Creating dynamic graphs")
            self._add_data_to_dynamic_graphs(is_multi)
        elif graph_type is "s":
            print("Creating static graph")
            self._add_data_to_static_graph(is_multi)
        else:
            raise Exception("ERROR - wrong graph value")
        print("Graphs created")

    def _add_data_to_static_graph(self, is_multi):
        """
        Adds data to static graph - all days.
        :param is_multi: bool
            True - multi-graph
            False - single-edges graph
        """
        graph = SocialNetworkGraph(is_multi=is_multi)
        for comments in self.comments_to_add:
            edges = numpy.concatenate(comments, axis=0)
            graph.add_edges(edges)
        graph.start_day, graph.end_day = self.days[0], self.days[-1]
        self.static_graph = graph

    def _add_data_to_dynamic_graphs(self, is_multi):
        """
        Adds data to dynamic graphs - selected days only.
        :param is_multi: bool
            True - multi-graph
            False - single-edges graph
        """
        graph = SocialNetworkGraph(is_multi=is_multi)
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
            graph = SocialNetworkGraph(is_multi=is_multi)
            i += step

    def create_graphs(self, mode):
        """
        Creates graphs corresponding to the selected mode.
        :param mode: Mode
            Defines model mode (which comments should me included)
        """
        self.mode = mode
        graphs_file_name = 'graphs' + "/" + self.mode.name \
            + "_" + str(self._number_of_days_in_interval) \
            + "_" + str(self._number_of_new_days_in_interval)
        Path('graphs').mkdir(parents=True, exist_ok=True)

        if os.path.exists(graphs_file_name):
            self.load_graphs_from_file(graphs_file_name)
            GraphIterator.set_graphs(self.static_graph, self.dynamic_graphs)
        else:
            self._read_salon24_comments_data_by_day()
            self._add_data_to_graphs("sd", False)
            GraphIterator.set_graphs(self.static_graph, self.dynamic_graphs)
            self.save_graphs_to_file(graphs_file_name)

    def load_graphs_from_file(self, graphs_file_name):
        print("Loading graphs from file")
        with open(graphs_file_name, 'rb') as file:
            dictionary = pickle.load(file)
            self.static_graph = dictionary['static']
            self.dynamic_graphs = dictionary['dynamic']

    def save_graphs_to_file(self, graphs_file_name):
        with open(graphs_file_name, 'wb') as file:
            pickle.dump({'static': self.static_graph, 'dynamic': self.dynamic_graphs}, file)

    def calculate(self, calculated_value,
                  save_to_file=True, save_to_database = True, predict=False, calculate_histogram=False,
                  x_scale=None, size_scale=None, data_condition_function=None, data_functions=None):
        """
        Calculates metrics values for each user and allows creating files and saving to database.
        :param save_to_database:
            True - save calculated value to database.
        :param calculated_value: MetricsType
            Calculate function from given class is called
        :param save_to_file: bool
            True - save full data to file
        :param predict: bool
            True - predict time series.
        :param calculate_histogram: bool
            True - calculate and save data as a histogram.
        :param x_scale: array (float)
            Defines standard classes for the histogram.
        :param size_scale: array (int)
            Defines size classes for the histogram.
        :param data_condition_function: function
            Condition function defines which values should be removed to e.g. remove None values
        :param data_functions: array (function)
            Data function defines how data should be modified in order to aggregate them e.g. minimum
        """
        file_writer = self.initialize_file_writer(save_to_file, calculated_value)
        histogram_managers = self.initialize_histogram_managers(calculate_histogram, data_functions, calculated_value,
                                                                x_scale, size_scale)
        bar = ProgressBar("Calculating %s (%s)" % (calculated_value.value, self.mode), "Calculated",
                          len(self.authors_ids))
        for i in range(len(self.authors_ids)):  # For each author (node)
            bar.next()
            first_activity_date = self.get_first_activity_date(self.authors_ids[i])
            data = calculated_value.calculate(self.authors_ids[i], first_activity_date)
            self.predict(predict, data, self.authors_names[i])
            data_modified = self.modify_data(data, data_condition_function) if calculate_histogram or save_to_file else []
            self.add_data_to_histograms(histogram_managers, self.authors_static_neighborhood_size[i], data_modified)
            self.save_data_to_file(file_writer, self.authors_ids[i], self.authors_names[i], data_modified)
            self.save_to_database(save_to_database, self.authors_ids[i], calculated_value, data_modified)

        self.save_histograms_to_file(str(self.mode.value), histogram_managers)
        bar.finish()

    def save_to_database(self, save_to_database, author_id, calculated_value, data):
        if save_to_database:
            self._databaseEngine.update_value_column(calculated_value.value, calculated_value.graph_iterator.get_mode(),
                                                     author_id, data)

    @staticmethod
    def save_histograms_to_file(mode_name, histogram_managers):
        for histogram in histogram_managers:
            histogram.save('output/' + mode_name + "/")

    @staticmethod
    def save_data_to_file(file_writer, author_id, author_name, data):
        if file_writer is not None:
            file_writer.write_row_to_file([author_id, author_name, *data])

    @staticmethod
    def add_data_to_histograms(histogram_managers, size, data):
        for h in histogram_managers:
            h.add_data(size, data)

    @staticmethod
    def modify_data(data, data_condition_function):
        data_modified = []
        for d in data:
            if d is not None and (data_condition_function is None or data_condition_function(d)):
                data_modified.append(d)
            else:
                data_modified.append('')
        return data_modified

    def initialize_histogram_managers(self, calculate_histogram, data_functions, calculated_value, x_scale, size_scale):
        if calculate_histogram:
            histogram_managers = []
            self.calculate_and_get_authors_static_neighborhood_size()
            for data_function in data_functions:
                histogram_managers.append(self.HistogramManager(data_function, calculated_value, x_scale, size_scale))
            return histogram_managers
        return []

    def initialize_file_writer(self, save_to_file, calculated_value):
        if save_to_file:
            file_name = calculated_value.get_name() + ".txt"
            file_writer = FileWriter()
            file_writer.set_all(self.mode.value, file_name, self.get_graph_labels(3))
            return file_writer
        return None

    def predict(self, predict, data, author_name):
        if predict:
            plot_data = []
            title_data = []
            interesting_ids = [1672, 440, 241, 2177, 797, 3621, 11, 2516]
            methods = [Prediction.exponential_smoothing, Prediction.ARIMA]
            parameters_versions = [i for i in range(3)]
            data = self.make_data_positive(diff(data))
            # print("Predict")
            prediction = Prediction(data, author_name)
            for parameters_version in parameters_versions:
                result = prediction.predict(0, len(data) - 50, 50, Prediction.exponential_smoothing,
                                            Prediction.MAPE_error, parameters_version)
                if len(plot_data) == 0:
                    plot_data.append((result[2].index, result[2]))
                    title_data.append("Original")
                plot_data.append(result[1])
                title_data.append(result[0] + str(parameters_version))
            if author_name in interesting_ids:
                Prediction.plot("Prediction for " + author_name, plot_data, title_data)

    @staticmethod
    def make_data_positive(data):
        minimum = min(data)
        return data + abs(minimum) + 1

    def get_authors(self, parameter):
        return [x[0] for x in self._databaseEngine.execute("SELECT " + parameter + " FROM authors ORDER BY id")]

    def get_graph_labels(self, empty_count=0):
        row_captions = [str(g.start_day) for g in self.dynamic_graphs]
        row_captions[0] = "static"
        for e in range(empty_count):
            row_captions.insert(0, ",")
        return row_captions

    def get_first_activity_date(self, author_id):
        try:
            return self._databaseEngine.execute("SELECT %s FROM authors WHERE id = %s"
                                                % ("first_activity_date", author_id))[0][0]
        except IndexError:
            return None

    def calculate_and_get_authors_static_neighborhood_size(self):
        if self.authors_static_neighborhood_size is None:
            calculated_value = MetricsType(MetricsType.NEIGHBORS_COUNT, GraphConnectionType.IN,
                                           GraphIterator(GraphIterator.GraphMode.STATIC))
            self.authors_static_neighborhood_size = \
                {i: calculated_value.calculate(self.authors_ids[i],
                                               self.get_first_activity_date(self.authors_ids[i]))[0]
                 for i in range(len(self.authors_ids))}
        return self.authors_static_neighborhood_size
