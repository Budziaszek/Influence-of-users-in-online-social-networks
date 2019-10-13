import datetime as dt
from collections import defaultdict
from ProgressBar import ProgressBar

from Data.PostgresDatabaseEngine import PostgresDatabaseEngine
from Network.SocialNetworkGraph import SocialNetworkGraph
from Data.FileWriter import FileWriter
from Mode import Mode
import statistics
import numpy


class Manager:
    _databaseEngine = PostgresDatabaseEngine()
    _number_of_days_in_interval = 7
    _number_of_new_days_in_interval = 3
    graph_data = []
    graph_type = None
    _comments_to_comments = []
    _comments_to_posts = []
    _comments_to_comments_from_others = []
    _comments_to_posts_from_others = []
    _comments_to_comments_exists = False
    _comments_to_posts_exists = False
    _comments_to_comments_from_others_exists = False
    _comments_to_posts_from_others_exists = False

    # Create DatabaseEngine that connects to database with given parameters and later
    # allow operations (e.g. queries). Define time intervals.
    def __init__(self, parameters):
        self._databaseEngine.connect(parameters)
        self._databaseEngine.create_first_activity_date_column()
        self._dates_range, self._days_count = self.__check_dates_range_and_count()
        self._comments_by_day_index, self.days = None, None
        self.mode = None
        # self._days_count = 7

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
            self._comments_to_posts.append(comments_to_posts)
        else:
            self._comments_to_posts_from_others.append(comments_to_posts)

    def __read_comments_to_comments(self, day_start, day_end, from_others=False):
        comments_to_comments = self._databaseEngine.execute("""SELECT c.author_id, p.author_id
                                                                        FROM comments c
                                                                        INNER JOIN comments p 
                                                                        ON c.parentcomment_id = p.id
                                                                        WHERE c.parentcomment_id IS NOT NULL
                                                                        AND c.date BETWEEN """
                                                        + "'" + str(day_start) + "' and '" + str(day_end) + "'"
                                                        + ("" if not from_others else "AND c.author_id!=p.author_id"))
        if not from_others:
            self._comments_to_comments.append(comments_to_comments)
        else:
            self._comments_to_comments_from_others.append(comments_to_comments)

    def __read_selected_types_of_comments(self, day_start, day_end):
        if not self._comments_to_posts_exists and self.mode.do_read_comments_to_posts:
            self.__read_comments_to_posts(day_start, day_end)
        if not self._comments_to_comments_exists and self.mode.do_read_comments_to_comments:
            self.__read_comments_to_comments(day_start, day_end)
        if not self._comments_to_posts_from_others_exists and self.mode.do_read_comments_to_posts_from_others:
            self.__read_comments_to_posts(day_start, day_end, True)
        if not self._comments_to_comments_from_others_exists and self.mode.do_read_comments_to_comments_from_others:
            self.__read_comments_to_comments(day_start, day_end, True)

    def __join_data_and_mark_calculated(self):
        if self.mode is Mode.comments_to_posts:
            self._comments_to_posts_exists = True
            return self._comments_to_posts
        if self.mode is Mode.comments_to_posts_from_others:
            self._comments_to_posts_from_others_exists = True
            return self._comments_to_posts_from_others

        if self.mode is Mode.comments_to_comments:
            self._comments_to_comments_exists = True
            return self._comments_to_comments
        if self.mode is Mode.comments_to_comments_from_others:
            self._comments_to_comments_exists_from_others = True
            return self._comments_to_comments_from_others

        if self.mode is Mode.comments_to_posts_and_comments:
            self._comments_to_posts_exists = True
            self._comments_to_comments_exists = True
            return [self._comments_to_posts[i] + self._comments_to_comments[i]
                    for i in range(len(self._comments_to_posts))]
        if self.mode is Mode.comments_to_posts_and_comments_from_others:
            self._comments_to_posts_from_others_exists = True
            self._comments_to_comments_from_others_exists = True
            return [self._comments_to_posts_from_others[i] + self._comments_to_comments_from_others[i]
                    for i in range(len(self._comments_to_posts_from_others))]

    # Retrieve the most important data about comments (tuple: (comment author, post author)) by day from database
    # and store data in array (chronologically day by day)
    def __read_salon24_comments_data_by_day(self, mode):
        print("Selecting data")
        self.mode = mode
        days = []
        bar = ProgressBar(self._days_count)

        for day in (self._dates_range[0] + dt.timedelta(n) for n in range(self._days_count)):
            # print("Select data for", day_start)
            bar.next()
            day_start = day.replace(hour=00, minute=00, second=00)
            day_end = day.replace(hour=23, minute=59, second=59)
            days.append(day_start.date())
            self.__read_selected_types_of_comments(day_start, day_end)
        bar.finish()
        print("Data selected")

        return self.__join_data_and_mark_calculated(), days

    @staticmethod
    def remove_authors_own_comment(comments_by_day_index):
        print("Removing authors own comments")
        bar = ProgressBar(len(comments_by_day_index))
        for comments in comments_by_day_index:
            bar.next()
            for comment in comments:
                if comment[0] == comment[1]:
                    comments.remove(comment)
        bar.finish()

    def add_data_to_graphs(self, graph_type, is_multi):
        self.graph_data = []
        self.set_graph_type(graph_type)
        if graph_type is "sd":
            print("Creating static graph and dynamic graphs")
            self.add_data_to_static_graph(is_multi)
            self.add_data_to_dynamic_graphs(is_multi)
        elif graph_type is "d":
            print("Creating dynamic graphs")
            self.add_data_to_dynamic_graphs(is_multi)
        elif graph_type is "s":
            print("Creating static graph")
            self.add_data_to_static_graph(is_multi)
        else:
            raise Exception("ERROR - wrong graph type")
        print("Graphs created")
        return graph_type

    def set_graph_type(self, graph_type):
        if self.graph_type is None:
            self.graph_type = graph_type

    def add_data_to_static_graph(self, is_multi):
        graph = SocialNetworkGraph(is_multi=is_multi)
        edges = self.__join_array_of_arrays(self._comments_by_day_index, 0)
        graph.add_edges(edges)
        self.graph_data.append(graph)
        # print("Graph number %s created: %s to %s (static)" % (
        #     self.graph_data.index(graph), self.days[0], self.days[-1]))

    def add_data_to_dynamic_graphs(self, is_multi):
        graph = SocialNetworkGraph(is_multi=is_multi)
        step = self._number_of_new_days_in_interval
        interval_length = self._number_of_days_in_interval
        i = 0
        while i + step <= self._days_count:
            end = i + interval_length - 1
            if end >= self._days_count:
                return
            edges = self.__join_array_of_arrays(self._comments_by_day_index, i, end + 1)
            graph.add_edges(edges)
            graph.start_day, graph.end_day = self.days[i], self.days[end]
            self.graph_data.append(graph)
            graph = SocialNetworkGraph(is_multi=is_multi)
            i += step

    def generate_graph_data(self, mode, graph_type, is_multi):
        self._comments_by_day_index, self.days = self.__read_salon24_comments_data_by_day(mode)
        self.add_data_to_graphs(graph_type, is_multi)

    def calculate_neighborhoods_multiply_values(self, calculated_values, connection_type):
        for calculated_value in calculated_values:
            self.calculate_neighborhoods(calculated_value, connection_type)

    def connections_strength_prepare_dict(self, calculated_value):
        neighborhoods_by_graph_by_size = []
        if calculated_value == "connections_strength":
            for g in range(len(self.graph_data)):
                neighborhoods_by_graph_by_size.append(defaultdict(list))
        return neighborhoods_by_graph_by_size

    def calculate_neighborhoods(self, calculated_value, connection_type):
        file_name = calculated_value + "_" + connection_type.value + ".txt"
        print("Creating file", file_name, "for connection type", "<" + connection_type.value + ">")
        file_writer = FileWriter()
        file_writer.set_all(self.mode.value, file_name, self.get_graph_labels(3))
        authors_names = self.get_authors("name")
        authors_ids = self.get_authors("id")
        neighborhoods_by_graph_by_size = self.connections_strength_prepare_dict(calculated_value)

        print("Calculating %s (%s %s)" % (calculated_value, connection_type, self.mode))
        bar = ProgressBar(len(authors_ids))
        for i in range(len(authors_ids)):  # For each author (node)
            bar.next()

            # Prepare labels for row
            data = [authors_ids[i], authors_names[i]]
            first_activity_date = self.get_first_activity_date(authors_ids[i])

            # Information about self-connection
            value = self.graph_data[0].has_edge(authors_ids[i], authors_ids[i])
            data.append(value)

            # Add value for each graph
            for g in range(len(self.graph_data)):
                graph = self.graph_data[g]
                value = None
                if not self.is_author_active(first_activity_date, graph.end_day):
                    value = ''
                else:
                    # Check which parameter to calculate was selected
                    if calculated_value == "neighbors_count":  # Check number of neighbors
                        value = connection_type.neighbors_count(graph, authors_ids[i])
                    if calculated_value == "connections_count":  # Check number of connections
                        value = connection_type.connections_count(graph, authors_ids[i])
                    if calculated_value == "connections_strength":  # Check neighborhood strength
                        value = connection_type.connections_strength(graph, authors_ids[i],
                                                                     neighborhoods_by_graph_by_size[g])
                    if calculated_value == "reciprocity":  # Check reciprocity
                        dictionary = graph.reciprocity([authors_ids[i]])
                        value = dictionary[authors_ids[i]] if authors_ids[i] in dictionary else 0
                # Append to row of data (about the current author)
                data.append(value)
            # Write row to file
            file_writer.write_row_to_file(data)
        bar.finish()
        if calculated_value == "connections_strength":
            self.connection_strength_measures(neighborhoods_by_graph_by_size, connection_type)
        print("Done")

    def connection_strength_measures(self, neighborhoods_by_graph_by_size, connection_type):
        m = self.get_max_key(neighborhoods_by_graph_by_size)
        s_range = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.calculate_and_save_measure(neighborhoods_by_graph_by_size, m, connection_type,
                                        "mean", statistics.mean)
        self.calculate_and_save_measure(neighborhoods_by_graph_by_size, m, connection_type,
                                        "minimum", min)
        self.calculate_and_save_measure(neighborhoods_by_graph_by_size, m, connection_type,
                                        "maximum", max)
        self.calculate_and_save_measure(neighborhoods_by_graph_by_size, m, connection_type,
                                        "median", statistics.median)
        self.calculate_and_save_measure(neighborhoods_by_graph_by_size, m, connection_type,
                                        "standard_deviation", statistics.stdev)
        self.calculate_and_save_measure(neighborhoods_by_graph_by_size, m, connection_type,
                                        "count_histogram", numpy.histogram, s_range)

    def calculate_and_save_measure(self, dictionary, m, connection_type, measure_name, measure_function, s_range=None):
        print("Calculating measure of connection strength %s (%s %s)" % (measure_name, connection_type, self.mode))
        bar = ProgressBar(m - 1)
        values = [[] for _ in range(0, m)]
        for i in range(2, m):  # For each neighborhood size (sorted!)
            bar.next()
            for g in range(len(self.graph_data)):  # For each graph
                if i in dictionary[g]:  # If size in data for this graph
                    array = dictionary[g][i]
                    if s_range is None:
                        if measure_function is statistics.stdev and len(array) < 2:
                            values[i - 2].append('')
                        else:
                            values[i - 2].append(measure_function(array))
                    else:
                        values[i - 2].append(measure_function(array, s_range))
                else:
                    values[i - 2].append('')
        bar.finish()
        self.save_array_do_file("strength_" + measure_name + "_by_range", connection_type.value, values, 0, m)

    @staticmethod
    def get_max_key(dictionary):
        m = 0
        for d in dictionary:
            n_m = max((int(k) for k in d.keys()), default=0)
            if n_m > m:
                m = n_m
        return m + 1

    def save_array_do_file(self, value_name, connection_type_value, array, minimum_row_i, maximum_row_i):
        file_name = str(value_name) + "_" + connection_type_value + ".txt"
        file_writer = FileWriter()
        file_writer.set_path(self.mode.value, file_name)
        file_writer.clean_file()
        file_writer.write_row_to_file(self.get_graph_labels(3))
        for i in range(minimum_row_i, maximum_row_i):
            file_writer.write_row_to_file([str(i)] + array[i - 2])

    @staticmethod
    def __join_array_of_arrays(array_of_arrays, start=0, end=None):
        if end is None:
            end = len(array_of_arrays) + 1
        return [j for i in array_of_arrays[start:end] for j in i]

    def get_authors(self, parameter):
        return [x[0] for x in self._databaseEngine.execute("SELECT " + parameter + " FROM authors ORDER BY id")]

    def get_graph_labels(self, empty_count=0):
        row_captions = []
        if self.graph_type == "d" or self.graph_type == "sd":
            row_captions = [str(g.start_day) for g in self.graph_data]
        if self.graph_type == "s":
            row_captions.insert(0, "static")
        if self.graph_type == "sd":
            row_captions[0] = "static"
        for e in range(empty_count):
            row_captions.insert(0, "")
        return row_captions

    def get_first_activity_date(self, author_id):
        try:
            return self._databaseEngine.execute("SELECT %s FROM authors WHERE id = %s"
                                                % ("first_activity_date", author_id))[0][0]
        except IndexError:
            return None

    @staticmethod
    def is_author_active(first_activity_date, end_day):
        if end_day is None:
            return True
        if first_activity_date is None:
            return False
        return first_activity_date <= end_day
