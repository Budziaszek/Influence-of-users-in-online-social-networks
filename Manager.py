import datetime as dt
from HistogramWithSize import HistogramWithSize
from ProgressBar import ProgressBar
from Data.PostgresDatabaseEngine import PostgresDatabaseEngine
from Network.SocialNetworkGraph import SocialNetworkGraph
from Data.FileWriter import FileWriter
from Mode import Mode
from MetricsType import MetricsType
import numpy


class Manager:
    _databaseEngine = PostgresDatabaseEngine()
    _number_of_days_in_interval = 7
    _number_of_new_days_in_interval = 3
    graph_data = []
    graph_type = None
    days = []
    comments_to_add = None
    mode = None
    authors_ids = None
    authors_names = None
    connection_type = None
    static_neighborhood_size = None

    # (Array, Does_Exist)
    _comments_to_comments = ([], False)
    _comments_to_posts = ([], False)
    _comments_to_comments_from_others = ([], False)
    _comments_to_posts_from_others = ([], False)

    @staticmethod
    def is_author_active(first_activity_date, end_day):
        if end_day is None:
            return True
        if first_activity_date is None:
            return False
        return first_activity_date <= end_day

    @staticmethod
    def get_static_neighbors_from_manager_instance(manager):
        return manager.get_authors_static_neighborhood_size()

    @staticmethod
    def get_max_key(dictionary):
        m = 0
        for d in dictionary:
            n_m = max((int(k) for k in d.keys()), default=0)
            if n_m > m:
                m = n_m
        return m + 1

    @staticmethod
    def __join_array_of_numpy_arrays(array_of_numpy_arrays, start=0, end=None):
        if end is None:
            end = len(array_of_numpy_arrays) + 1
        return numpy.concatenate(array_of_numpy_arrays[start:end], axis=0)

    # Create DatabaseEngine that connects to database with given parameters and later
    # allow operations (e.g. queries). Define time intervals.
    def __init__(self, parameters):
        self._databaseEngine.connect(parameters)
        self._databaseEngine.create_first_activity_date_column()
        self._dates_range, self._days_count = self.__check_dates_range_and_count()
        self.authors_ids = self.get_authors("id")
        self.authors_names = self.get_authors("name")
        # self._days_count = 20

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
                                                            + (
                                                                "" if not from_others else "AND c.author_id!=p.author_id"))
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
        self._mark_calculated()
        return days

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
            raise Exception("ERROR - wrong graph value")
        print("Graphs created")
        return graph_type

    def set_graph_type(self, graph_type):
        if self.graph_type is None:
            self.graph_type = graph_type

    def add_data_to_static_graph(self, is_multi):
        graph = SocialNetworkGraph(is_multi=is_multi)
        for comments in self.comments_to_add:
            edges = self.__join_array_of_numpy_arrays(comments, 0)
            graph.add_edges(edges)
        self.graph_data.append(graph)

    def add_data_to_dynamic_graphs(self, is_multi):
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
            self.graph_data.append(graph)
            graph = SocialNetworkGraph(is_multi=is_multi)
            i += step

    def generate_graph_data(self, mode, graph_type, is_multi):
        self.days = self.__read_salon24_comments_data_by_day(mode)
        self.add_data_to_graphs(graph_type, is_multi)
        self.static_neighborhood_size = None

    def calculate(self, calculated_value, connection_type, calculate_full_data=True, calculate_histogram=False,
                  x_scale=None, size_scale=None, data_condition_function=None, data_function=None, mode=None):
        file_writer, hist, authors_static_neighborhood_size, file_name_hist = None, None, None, None
        self.connection_type = connection_type

        if calculate_full_data:
            file_name = calculated_value.value + "_" + self.connection_type.value + ".txt"
            print("Calculating %s (%s %s)" % (calculated_value.value, self.connection_type.value, self.mode))
            print("Creating file", file_name, "for connection value", "<" + self.connection_type.value + ">")
            file_writer = FileWriter()
            file_writer.set_all(self.mode.value, file_name, self.get_graph_labels(3))

        if calculate_histogram:
            if data_function is None:
                fun_name = "all"
            else:
                fun_name = str(data_function.__name__)
            file_name_hist = mode + "_" + fun_name + "_" + "histogram_" + calculated_value.value + "_" \
                             + self.connection_type.value + ".txt"
            print("Calculating histogram %s (%s %s)" % (calculated_value.value,
                                                        self.connection_type.value, self.mode))
            print("Creating file", file_name_hist, "for connection value", "<" + self.connection_type.value + ">")
            hist = HistogramWithSize(x_scale, size_scale)
            authors_static_neighborhood_size = self.get_authors_static_neighborhood_size()

        bar = ProgressBar(len(self.authors_ids))
        for i in range(len(self.authors_ids)):  # For each author (node)
            bar.next()
            # Prepare labels for row
            data = []
            data_hist = []
            first_activity_date = self.get_first_activity_date(self.authors_ids[i])
            # Add value for each graph
            for g in range(len(self.graph_data)):
                graph = self.graph_data[g]
                if not self.is_author_active(first_activity_date, graph.end_day):
                    data.append('')
                else:
                    value = calculated_value.calculate(self.connection_type, self.graph_data, g,
                                                       self.authors_ids[i], self)
                    # Append to row of data (about the current author)
                    if data_condition_function is not None and data_condition_function(value):
                        data.append(value)
                        if mode == "s" and g == 0:
                            data_hist.append(value)
                        elif mode == "d" and g != 0:
                            data_hist.append(value)
                    else:
                        data.append('')
            # Add data to histogram
            if calculate_histogram:
                hist.add(authors_static_neighborhood_size[i], data_hist, data_function)
            # Write row to file
            if calculate_full_data:
                file_writer.write_row_to_file([self.authors_ids[i], self.authors_names[i], *data])
        if calculate_histogram:
            hist.save('output/' + str(self.mode.value) + "/", file_name_hist)
        bar.finish()
        print("Done")

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
        row_captions = []
        if self.graph_type == "d" or self.graph_type == "sd":
            row_captions = [str(g.start_day) for g in self.graph_data]
        if self.graph_type == "s":
            row_captions.insert(0, "static")
        if self.graph_type == "sd":
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
        if self.static_neighborhood_size is None:
            if not (self.graph_type is "sd" or self.graph_type is "s"):
                raise Exception("No static graph included")
            else:
                calculated_value = MetricsType(MetricsType.NEIGHBORS_COUNT)
                self.static_neighborhood_size = {i: calculated_value.calculate(self.connection_type, self.graph_data,
                                                                               0, self.authors_ids[i], self)
                                                 for i in range(len(self.authors_ids))}
        return self.static_neighborhood_size
