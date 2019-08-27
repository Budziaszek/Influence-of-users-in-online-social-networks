import datetime as dt
from collections import defaultdict

from Data.PostgresDatabaseEngine import PostgresDatabaseEngine
from Network.SocialNetworkGraph import SocialNetworkGraph
from Data.FileWriter import FileWriter


class Manager:
    _databaseEngine = PostgresDatabaseEngine()
    _number_of_days_in_interval = 7
    _number_of_new_days_in_interval = 3
    graph_data = []
    file_writer = FileWriter()
    mode = None

    # Create DatabaseEngine that connects to database with given parameters and later
    # allow operations (e.g. queries). Define time intervals.
    def __init__(self, parameters):
        self._databaseEngine.connect(parameters)
        self._databaseEngine.create_first_activity_date_column()
        self._dates_range, self._days_count = self.__check_dates_range_and_count()
        # self._days_count = 50
        self._comments_by_day_index, self.days = self.__read_salon24_comments_data_by_day()

    # Check dates range (find min and max comment time)
    def __check_dates_range_and_count(self, date_row_name="date", analyzed_column_name="comments"):
        dates_range = self._databaseEngine.execute("SELECT min(" + date_row_name + "), max(" + date_row_name
                                                   + ") FROM " + analyzed_column_name).pop()
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
            days.append(day_start)
            comments_by_day_index.append(self._databaseEngine
                                         .execute("""SELECT c.author_id, p.author_id 
                                                FROM comments c
                                                INNER JOIN posts p ON c.post_id = p.id
                                                WHERE c.date BETWEEN %s and %s""",
                                                  (day_start, day_end)))
            print("Select data for", day_start)
        return comments_by_day_index, days

    def generate_graph_data(self, mode, is_multi):
        graph = SocialNetworkGraph(is_multi=is_multi)
        if self.mode is None:
            self.mode = mode
        i = 0
        # First graph is static, next ones dynamic
        if mode is "sd":
            self.generate_graph_data("s", is_multi)
            self.generate_graph_data("d", is_multi)
            return
        # Dynamic graphs - Create graphs for intervals. Each interval has 7 days, next interval has
        # 4 days common with previous one. Note that las interval can be shorter.
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

        while i + step <= self._days_count:
            end = i + interval_length
            end = end if end < self._days_count else self._days_count  # Include last interval (which is shorter)
            edges = self.__join_array_of_arrays(self._comments_by_day_index, i, end)
            graph.add_edges(edges)
            # print("{0}-{1} count: {2}".format(i, end - 1, len(edges)))  # Including the last element so -1 added
            if step is 1:  # Static graph - update graph data
                self.graph_data.clear()
                self.graph_data.append(graph)
                print("Data (day %s) added to the graph %s" % (i, self.graph_data.index(graph)))
            else:  # Dynamic graphs - append graph and create new for next interval
                graph.start_day = self.days[i]
                graph.end_day = self.days[end - 1]
                self.graph_data.append(graph)
                print("Graph number %s created" % self.graph_data.index(graph))
                graph = SocialNetworkGraph(is_multi)
            i += step

    def calculate_neighborhoods(self, calculated_value, connection_type):
        file_name = calculated_value + "_" + connection_type.value + ".txt"
        print("Creating file", file_name, "for connection type", "<" + connection_type.value + ">")
        self.file_writer.set_file(file_name)
        self.file_writer.clean_file()

        if calculated_value == "connections_strength":
            neighborhoods_by_size = defaultdict(list)

        authors_names = self.get_authors("name")
        authors_ids = self.get_authors("id")
        for i in range(len(authors_ids)):  # For each author (node)
            print("User", i, "/", len(authors_ids), authors_names[i])
            # Prepare labels for row
            data = [authors_ids[i], authors_names[i]]
            first_activity_date = self.get_first_activity_date(authors_ids[i])
            for graph in self.graph_data:  # For each graph
                value = None
                if not self.is_author_active(first_activity_date, graph.end_day):
                    value = ''
                else:
                    # Check which parameter to calculate was selected
                    if calculated_value == "neighbors_count":  # Check number of neighbors
                        value = connection_type.neighbors_count(graph, authors_ids[i])
                    if calculated_value == "connections_count":  # Check number of connections
                        value = connection_type.connections_count(graph, authors_ids[i])
                    if calculated_value == "connections_strength":
                        value = connection_type.connections_strength(graph, authors_ids[i], neighborhoods_by_size)
                # Append to row of data (about the current author)
                data.append(value)
            # Write row to file
            self.file_writer.write_row_to_file(data)
        if calculated_value == "connections_strength":
            open("dict" + connection_type.value + ".txt", 'w').close()
            with open("dict" + connection_type.value + ".txt", "a+", encoding="utf-8") as f:
                for key, value in neighborhoods_by_size.items():
                    f.write(str(key) + ", " + str(value) + "")
                    f.write("\n")
        print("Done")

    @staticmethod
    def __join_array_of_arrays(array_of_arrays, start=0, end=None):
        if end is None:
            end = len(array_of_arrays)
        return [j for i in array_of_arrays[start:end] for j in i]

    def get_authors(self, parameter):
        return [x[0] for x in self._databaseEngine.execute("SELECT " + parameter + " FROM authors ORDER BY id")]

    def get_graph_labels(self):
        row_captions = []
        if self.mode == "d" or self.mode == "sd":
            row_captions = ["dynamic " + str(i) for i in range(len(self.graph_data))]
        if self.mode == "s" or self.mode == "sd":
            row_captions.insert(0, "static")
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
        return first_activity_date <= end_day.date()
