import datetime as dt
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
    _number_of_days_in_interval = 60  # 7
    _number_of_new_days_in_interval = 30  # 3
    dynamic_graphs = []
    static_graph = None
    days = []
    comments_to_add = None
    mode = None
    authors_ids = None
    authors_names = None

    # (Array, Does_Exist)
    _comments_to_comments = ([], False)
    _comments_to_posts = ([], False)
    _comments_to_comments_from_others = ([], False)
    _comments_to_posts_from_others = ([], False)

    @staticmethod
    def __join_array_of_numpy_arrays(array_of_numpy_arrays, start=0, end=None):
        if end is None:
            end = len(array_of_numpy_arrays) + 1
        return numpy.concatenate(array_of_numpy_arrays[start:end], axis=0)

    # Create DatabaseEngine that connects to database with given parameters and later
    # allow operations (e.g. queries). Define time intervals.
    def __init__(self, parameters, test=False):
        self._databaseEngine.connect(parameters)
        self._databaseEngine.create_first_activity_date_column()
        self._dates_range, self._days_count = self.__check_dates_range_and_count()
        self.authors_ids = self.get_authors("id")
        self.authors_names = self.get_authors("name")
        self.authors_static_neighborhood_size = None
        if test is True:
            self._days_count = 40

    # Check dates range (find min and max comment time)
    def __check_dates_range_and_count(self):
        date_row_name = "date"
        analyzed_column_name_1 = "comments"
        analyzed_column_name_2 = "posts"
        dates_range_1 = self._databaseEngine.execute("SELECT min(" + date_row_name + "), max(" + date_row_name
                                                     + ") FROM " + analyzed_column_name_1).pop()
        dates_range_2 = self._databaseEngine.execute("SELECT min(" + date_row_name + "), max(" + date_row_name
                                                     + ") FROM " + analyzed_column_name_2).pop()
        dates_range = [min(dates_range_1[0], dates_range_2[0]), max(dates_range_1[1], dates_range_2[1])]
        # TODO programmatically split
        # dates_range = [dates_range[0], dates_range[0] + (dates_range[1] - dates_range[0])/2]
        # dates_range = [dates_range[0] + (dates_range[1] - dates_range[0]) / 2 - dt.timedelta(7), dates_range[1]]
        print("Dates range:", dates_range)
        return dates_range, (dates_range[1] - dates_range[0]).days

    def __read_comments_to_posts(self, day_start, day_end, from_others=False):
        comments_to_posts = self._databaseEngine.execute("""SELECT c.author_id, p.author_id 
                                                                        FROM comments c
                                                                        INNER JOIN posts p 
                                                                        ON c.post_id = p.id 
                                                                        WHERE c.date BETWEEN """
                                                         + "'" + str(day_start) + "' and '" + str(day_end) + "'"
                                                         + ("" if not from_others else "AND c.author_id!=p.author_id"))
        if not from_others:
            self._comments_to_posts[0].append(numpy.array(comments_to_posts))
        else:
            self._comments_to_posts_from_others[0].append(numpy.array(comments_to_posts))

    def __read_comments_to_comments(self, day_start, day_end, from_others=False):
        comments_to_comments = self._databaseEngine.execute("""SELECT c.author_id, p.author_id
                                                                        FROM comments c
                                                                        INNER JOIN comments p 
                                                                        ON c.parentcomment_id = p.id
                                                                        WHERE c.parentcomment_id IS NOT NULL
                                                                        AND c.date BETWEEN """
                                                            + "'" + str(day_start) + "' and '" + str(day_end) + "'"
                                                            + ("" if not from_others
                                                               else "AND c.author_id!=p.author_id"))
        if not from_others:
            self._comments_to_comments[0].append(numpy.array(comments_to_comments))
        else:
            self._comments_to_comments_from_others[0].append(numpy.array(comments_to_comments))

    def __read_selected_types_of_comments(self, day_start, day_end):
        if not self._comments_to_posts[1] and self.mode.do_read_comments_to_posts:
            self.__read_comments_to_posts(day_start, day_end)
        if not self._comments_to_comments[1] and self.mode.do_read_comments_to_comments:
            self.__read_comments_to_comments(day_start, day_end)
        if not self._comments_to_posts_from_others[1] and self.mode.do_read_comments_to_posts_from_others:
            self.__read_comments_to_posts(day_start, day_end, True)
        if not self._comments_to_comments_from_others[1] and self.mode.do_read_comments_to_comments_from_others:
            self.__read_comments_to_comments(day_start, day_end, True)

    def _mark_calculated(self):
        self.comments_to_add = []
        if self.mode is Mode.COMMENTS_TO_POSTS:
            self._comments_to_posts = (self._comments_to_posts[0], True)
            self.comments_to_add.append(self._comments_to_posts[0])
        if self.mode is Mode.COMMENTS_TO_POSTS_FROM_OTHERS:
            self._comments_to_posts_from_others = (self._comments_to_posts_from_others[0], True)
            self.comments_to_add.append(self._comments_to_posts_from_others[0])
        if self.mode is Mode.COMMENTS_TO_COMMENTS:
            self._comments_to_comments = (self._comments_to_comments[0], True)
            self.comments_to_add.append(self._comments_to_comments[0])
        if self.mode is Mode.COMMENTS_TO_COMMENTS_FROM_OTHERS:
            self._comments_to_comments_from_others = (self._comments_to_comments_from_others[0], True)
            self.comments_to_add.append(self._comments_to_comments_from_others[0])
        if self.mode is Mode.COMMENTS_TO_POSTS_AND_COMMENTS:
            self._comments_to_posts = (self._comments_to_posts[0], True)
            self._comments_to_comments = (self._comments_to_comments[0], True)
            self.comments_to_add.append(self._comments_to_posts[0])
            self.comments_to_add.append(self._comments_to_comments[0])
        if self.mode is Mode.COMMENTS_TO_POSTS_AND_COMMENTS_FROM_OTHERS:
            self._comments_to_posts_from_others = (self._comments_to_posts_from_others[0], True)
            self._comments_to_comments_from_others = (self._comments_to_comments_from_others[0], True)
            self.comments_to_add.append(self._comments_to_posts_from_others[0])
            self.comments_to_add.append(self._comments_to_comments_from_others[0])

    # Retrieve the most important statistics_values about comments (tuple: (comment author, post author))
    # by day from database and store statistics_values in array (chronologically day by day)
    def __read_salon24_comments_data_by_day(self, mode):
        print("Selecting data")
        self.mode = mode
        days = []
        bar = ProgressBar(self._days_count)

        for day in (self._dates_range[0] + dt.timedelta(n) for n in range(self._days_count)):
            # print("Select statistics_values for", day_start)
            bar.next()
            day_start = day.replace(hour=00, minute=00, second=00)
            day_end = day.replace(hour=23, minute=59, second=59)
            days.append(day_start.date())
            self.__read_selected_types_of_comments(day_start, day_end)
        bar.finish()
        print("Data selected")
        self._mark_calculated()
        return days

    def _add_data_to_graphs(self, graph_type, is_multi):
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
        return graph_type

    def _add_data_to_static_graph(self, is_multi):
        graph = SocialNetworkGraph(is_multi=is_multi)
        for comments in self.comments_to_add:
            edges = self.__join_array_of_numpy_arrays(comments, 0)
            graph.add_edges(edges)
        self.static_graph = graph

    def _add_data_to_dynamic_graphs(self, is_multi):
        graph = SocialNetworkGraph(is_multi=is_multi)
        step = self._number_of_new_days_in_interval
        interval_length = self._number_of_days_in_interval
        i = 0
        while i + step <= self._days_count:
            end = i + interval_length - 1
            if end >= self._days_count:
                return
            for comments in self.comments_to_add:
                # for j in range(i, end + 1):
                #     graph.add_edges(comments[j])
                edges = self.__join_array_of_numpy_arrays(comments, i, end + 1)
                graph.add_edges(edges)
            graph.start_day, graph.end_day = self.days[i], self.days[end]
            self.dynamic_graphs.append(graph)
            graph = SocialNetworkGraph(is_multi=is_multi)
            i += step

    def generate_graph_data(self, mode):
        self.days = self.__read_salon24_comments_data_by_day(mode)
        self._add_data_to_graphs("sd", False)
        GraphIterator.set_graphs(self.static_graph, self.dynamic_graphs)

    def calculate(self, calculated_value, calculate_full_data=True, predict=False, calculate_histogram=False,
                  x_scale=None, size_scale=None, data_condition_function=None, data_functions=None):
        file_writer = None

        if calculate_full_data:
            file_name = calculated_value.get_name() + ".txt"
            print("Calculating %s (%s)" % (calculated_value.value, self.mode))
            print("Creating file", file_name)
            file_writer = FileWriter()
            file_writer.set_all(self.mode.value, file_name, self.get_graph_labels(3))

        histograms = []
        if calculate_histogram:
            self.get_authors_static_neighborhood_size()
            for data_function in data_functions:
                if data_function is None:
                    fun_name = "all"
                else:
                    fun_name = str(data_function.__name__)
                print("Creating histogram for ", fun_name)
                file_name = fun_name + "_" + "hist_" + calculated_value.get_name() + ".txt"
                hist = Histogram(x_scale, size_scale)
                histograms.append([data_function, file_name, hist])

        bar = ProgressBar(len(self.authors_ids))
        for i in range(len(self.authors_ids)):  # For each author (node)
            bar.next()
            first_activity_date = self.get_first_activity_date(self.authors_ids[i])
            data = calculated_value.calculate(self.authors_ids[i], first_activity_date)

            plot_data = []
            title_data = []
            interesting_ids = [1672, 440, 241, 2177, 797, 3621, 11, 2516]
            methods = [Prediction.exponential_smoothing, Prediction.ARIMA]
            parameters_versions = [i for i in range(3)]
            data = self.make_positive(diff(data))
            if predict:
                print("Predict")
                prediction = Prediction(data, self.authors_names[i])
                for parameters_version in parameters_versions:
                    result = prediction.predict(0, len(data) - 50, 50, Prediction.exponential_smoothing,
                                                Prediction.MAPE_error, parameters_version)
                    if len(plot_data) == 0:
                        plot_data.append((result[2].index, result[2]))
                        title_data.append("Original")
                    plot_data.append(result[1])
                    title_data.append(result[0] + str(parameters_version))
                if self.authors_ids[i] in interesting_ids:
                    Prediction.plot("Prediction for " + self.authors_names[i], plot_data, title_data)

            full_data = []
            data_hist = []
            for d in data:
                if d is not None:
                    if data_condition_function is None or data_condition_function(d):
                        if d is not "":
                            data_hist.append(d)
                        full_data.append(d)
                    else:
                        full_data.append('')
                else:
                    full_data.append('')
            # Add statistics_values to histogram
            # [data_function, file_name, hist]
            if calculate_histogram:
                for histogram in histograms:
                    histogram[2].add(self.authors_static_neighborhood_size[i], data_hist, histogram[0])
            # Write row to file
            if calculate_full_data:
                file_writer.write_row_to_file([self.authors_ids[i], self.authors_names[i], *full_data])
        if calculate_histogram:
            for histogram in histograms:
                histogram[2].save('output/' + str(self.mode.value) + "/", histogram[1])
        bar.finish()
        print("Done")

    @staticmethod
    def make_positive(data):
        minimum = min(data)
        return data + abs(minimum) + 1

    def save_array_do_file(self, value_name, connection_type_value, array, minimum_row_i, maximum_row_i):
        file_name = str(value_name) + "_" + connection_type_value + ".txt"
        file_writer = FileWriter()
        file_writer.set_path(self.mode.value, file_name)
        file_writer.clean_file()
        file_writer.write_row_to_file(self.get_graph_labels(3))
        for i in range(minimum_row_i, maximum_row_i):
            file_writer.write_row_to_file([str(i)] + array[i - 2])

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

    def get_authors_static_neighborhood_size(self):
        if self.authors_static_neighborhood_size is None:
            calculated_value = MetricsType(MetricsType.NEIGHBORS_COUNT, GraphConnectionType.IN,
                                           GraphIterator(GraphIterator.GraphMode.STATIC))
            self.authors_static_neighborhood_size = \
                {i: calculated_value.calculate(self.authors_ids[i],
                                               self.get_first_activity_date(self.authors_ids[i]))[0]
                 for i in range(len(self.authors_ids))}
        return self.authors_static_neighborhood_size
