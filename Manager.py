import datetime as dt
from collections import defaultdict

from Data.PostgresDatabaseEngine import PostgresDatabaseEngine
from Network.SocialNetworkGraph import SocialNetworkGraph
from Data.FileWriter import FileWriter
import statistics
import numpy


class Manager:
    _databaseEngine = PostgresDatabaseEngine()
    _number_of_days_in_interval = 7
    _number_of_new_days_in_interval = 3
    graph_data = []
    mode = None

    # Create DatabaseEngine that connects to database with given parameters and later
    # allow operations (e.g. queries). Define time intervals.
    def __init__(self, parameters):
        self._databaseEngine.connect(parameters)
        self._databaseEngine.create_first_activity_date_column()
        self._dates_range, self._days_count = self.__check_dates_range_and_count()
        # self._days_count = 7
        self._comments_by_day_index, self.days = self.__read_salon24_comments_data_by_day()

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

    # Retrieve the most important data about comments (tuple: (comment author, post author)) by day from database
    # and store data in array (chronologically day by day)
    def __read_salon24_comments_data_by_day(self):
        comments_by_day_index = []
        days = []
        for day in (self._dates_range[0] + dt.timedelta(n) for n in range(self._days_count)):
            day_start = day.replace(hour=00, minute=00, second=00)
            day_end = day.replace(hour=23, minute=59, second=59)
            days.append(day_start.date())
            comments_by_day_index.append(self._databaseEngine
                                         .execute("""SELECT c.author_id, p.author_id 
                                                FROM comments c
                                                INNER JOIN posts p ON c.post_id = p.id
                                                WHERE c.date BETWEEN %s and %s""",
                                                  (day_start, day_end)))
            # With parent comment
            # comments_by_day_index.append(self._databaseEngine
            #                              .execute("""SELECT c.author_id, p.author_id
            #                                     FROM comments c
            #                                     INNER JOIN comments p ON c.parentcomment_id = p.id
            #                                     WHERE parentcomment_id IS NOT NULL
            #                                     AND c.date BETWEEN %s and %s""",
            #                                       (day_start, day_end)))
            print("Select data for", day_start)
        return comments_by_day_index, days

    def check_mode(self, mode, is_multi):
        # First graph is static, next ones dynamic
        if mode is "sd":
            self.generate_graph_data("s", is_multi)
            # self.generate_graph_data("d", is_multi)
            mode = "d"
        # Dynamic graphs - Create graphs for intervals. Each interval has 7 days, next interval has
        # 4 days common with previous one.
        if mode is "d":
            step = self._number_of_new_days_in_interval
            interval_length = self._number_of_days_in_interval
            print("Creating dynamic graphs")
        # Static graph for all data - One graph, actualize one and only graph with index 0
        elif mode is "s":
            step = 1
            interval_length = step
            print("Creating static graph")
        else:
            raise Exception("ERROR - wrong graph mode")
        return step, interval_length

    def generate_graph_data(self, mode, is_multi):
        graph = SocialNetworkGraph(is_multi=is_multi)
        if self.mode is None:
            self.mode = mode
        step, interval_length = self.check_mode(mode, is_multi)

        if step is 1:  # Static graph - update graph data
            edges = self.__join_array_of_arrays(self._comments_by_day_index, 0)
            graph.add_edges(edges)
            self.graph_data.append(graph)
            print("Graph number %s created: %s to %s (static)" % (self.graph_data.index(graph), self.days[0], self.days[-1]))
            return

        i = 0
        while i + step <= self._days_count:
            end = i + interval_length - 1
            shorter = end >= self._days_count
            end = end if not shorter else self._days_count  # Include (or not) last interval (which is shorter)
            edges = self.__join_array_of_arrays(self._comments_by_day_index, i, end + 1)
            graph.add_edges(edges)
            if not shorter:  # Dynamic graphs - append graph and create new for next interval, do not include shorter
                graph.start_day = self.days[i]
                graph.end_day = self.days[end]
                self.graph_data.append(graph)
                print("Graph number %s created: %s to %s (dynamic)" % (self.graph_data.index(graph), graph.start_day, graph.end_day))
                graph = SocialNetworkGraph(is_multi)
            i += step

    def calculate_neighborhoods(self, calculated_value, connection_type):
        neighborhoods_by_graph_by_size = None
        file_name = calculated_value + "_" + connection_type.value + ".txt"
        print("Creating file", file_name, "for connection type", "<" + connection_type.value + ">")
        file_writer = FileWriter()
        file_writer.set_file(file_name)
        file_writer.clean_file()
        file_writer.write_row_to_file(self.get_graph_labels())

        authors_names = self.get_authors("name")
        authors_ids = self.get_authors("id")

        if calculated_value == "connections_strength":
            neighborhoods_by_graph_by_size = []
            for g in range(len(self.graph_data)):
                neighborhoods_by_graph_by_size.append(defaultdict(list))
                open("strength_by_graph_by_size_" + str(g)
                     + "_" + connection_type.value + ".txt", 'w').close()

        for i in range(len(authors_ids)):  # For each author (node)
        # for i in range(100):  # For each author (node)
            print("User", i, "/", len(authors_ids), authors_names[i].encode("utf-8"))
            # Prepare labels for row
            data = [authors_ids[i], authors_names[i]]
            first_activity_date = self.get_first_activity_date(authors_ids[i])
            for g in range(len(self.graph_data)):  # For each graph
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
                # Append to row of data (about the current author)
                data.append(value)
            # Write row to file
            file_writer.write_row_to_file(data)

        if calculated_value == "connections_strength":
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
        print("Done")

    def calculate_and_save_measure(self, dictionary, m, connection_type, measure_name, measure_function, s_range=None):
        values = [[] for _ in range(0, m)]
        for i in range(2, m):  # For each neighborhood size (sorted!)
            for g in range(len(self.graph_data)):  # For each graph
                if i in dictionary[g]:  # If size in data for this graph
                    array = dictionary[g][i]
                    if s_range is None:
                        if measure_function is statistics.stdev and len(array)<2:
                            values[i - 2].append('')
                        else:
                            values[i - 2].append(measure_function(array))
                    else:
                        values[i - 2].append(measure_function(array, s_range))
                else:
                    values[i - 2].append('')
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
        file_writer.set_file(file_name)
        file_writer.clean_file()
        file_writer.write_row_to_file(self.get_graph_labels())
        for i in range(minimum_row_i, maximum_row_i):
            file_writer.write_row_to_file([str(i)] + array[i-2])
        # open(file_name, 'w').close()
        # f = open(file_name, 'w')
        # for i in range(minimum_row_i, maximum_row_i):
        #     f.write(str(i) + ", " + ', '.join(str(x) for x in array[i - 2]) + "\n")

    @staticmethod
    def __join_array_of_arrays(array_of_arrays, start=0, end=None):
        if end is None:
            end = len(array_of_arrays) + 1
        print(start, "-", end)
        return [j for i in array_of_arrays[start:end] for j in i]

    def get_authors(self, parameter):
        return [x[0] for x in self._databaseEngine.execute("SELECT " + parameter + " FROM authors ORDER BY id")]

    def get_graph_labels(self):
        row_captions = ["Labels"]
        if self.mode == "d" or self.mode == "sd":
            row_captions = [str(g.start_day) for g in self.graph_data]
        if self.mode == "s" or self.mode == "sd":
            row_captions[0] = "static"
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
