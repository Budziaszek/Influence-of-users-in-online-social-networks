import datetime as dt
from Database.PostgresDatabaseEngine import PostgresDatabaseEngine
from Network.NetworkEngine import NetworkEngine


class Manager:
    _databaseEngine = PostgresDatabaseEngine()
    _number_of_days_in_interval = 7
    _number_of_new_days_in_interval = 3
    graph_data = None

    # Create DatabaseEngine that connects to database with given parameters and later
    # allow operations (e.g. queries). Define time intervals.
    def __init__(self, parameters):
        self._databaseEngine.connect(parameters)
        self._dates_range, self._days_count = self.__check_dates_range_and_count()
        self._comments_by_day_index = self.__read_salon24_comments_data_by_day()

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
        for day in (self._dates_range[0] + dt.timedelta(n) for n in range(self._days_count)):
            day_start = day.replace(hour=00, minute=00, second=00)
            day_end = day.replace(hour=23, minute=59, second=59)
            comments_by_day_index.append(self._databaseEngine
                                         .execute("""SELECT c.author_id, p.author_id 
                                                FROM comments c
                                                INNER JOIN posts p ON c.post_id = p.id
                                                WHERE c.date BETWEEN %s and %s""",
                                                  (day_start, day_end)))
            print("Select data for", day_start)
        return comments_by_day_index

    def generate_graph_data(self, mode, is_multi):
        self.graph_data = []
        graph = NetworkEngine(is_multi=is_multi)
        i = 0
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
            print("ERROR - wrong graph mode")
            return

        while i + step <= self._days_count:
            end = i + interval_length
            end = end if end < self._days_count else self._days_count  # Include last interval (which is shorter)
            edges = self.__join_array_of_arrays(self._comments_by_day_index, i, end)
            graph.add_edges(edges)
            print("{0}-{1} count: {2}".format(i, end - 1, len(edges)))  # Including the last element so -1 added
            if step is 1:
                self.graph_data.clear()
                self.graph_data.append(graph)
            else:
                self.graph_data.append(graph)
                graph = NetworkEngine()
            i += step

    @staticmethod
    def __join_array_of_arrays(array_of_arrays, start=0, end=None):
        if end is None:
            end = len(array_of_arrays)
        return [j for i in array_of_arrays[start:end] for j in i]
