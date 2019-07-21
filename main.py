import datetime as dt

from Database.PostgresDatabaseEngine import PostgresDatabaseEngine
from Network.NetworkEngine import NetworkEngine

# Create DatabaseEngine that connects to database with given parameters and later allow operations (e.g. queries)
databaseEngine = PostgresDatabaseEngine()
parameters = "dbname='salon24' user='sna_user' host='localhost' password='sna_password'"
databaseEngine.connect(parameters)

# Check date range (find min and max comment time)
date_range = databaseEngine.execute("""SELECT min(date), max(date) FROM comments""").pop()
print("Date range:", date_range)

# Define time intervals
day_count = (date_range[1] - date_range[0]).days
# day_count = 7  # TODO remove (for tests only)
number_of_days_in_interval = 7
number_of_new_days_in_interval = 3

# Retrieve the most important data about comments (tuple: (comment author, post author)) by day from database
# and store data in array (chronologically day by day)
comments_edges_by_day_index = []
for day in (date_range[0] + dt.timedelta(n) for n in range(day_count)):
    day_start = day.replace(hour=00, minute=00, second=00)
    day_end = day.replace(hour=23, minute=59, second=59)
    comments_edges_by_day_index.append(databaseEngine.execute("""SELECT c.author_id, p.author_id
                                        FROM comments c
                                        INNER JOIN posts p ON c.post_id = p.id
                                        WHERE c.date BETWEEN %s and %s""",
                                                              (day_start, day_end)))

# Create graphs for intervals. Each interval has 7 days, next interval has 4 days common with previous one.
# Note that las interval can be shorter.
i = 0
while i < day_count:
    network = NetworkEngine()
    if i + number_of_days_in_interval < day_count:
        edges_for_time_interval = [j for i in comments_edges_by_day_index[i:i + number_of_days_in_interval] for j in i]
        # # Including the last element so -1 added
        print("{0}-{1} count: {2}".format(i, i + number_of_days_in_interval - 1, len(edges_for_time_interval)))
    else:  # Last interval that is shorter
        edges_for_time_interval = [j for i in comments_edges_by_day_index[i:day_count] for j in i]
        print("{0}-{1} count: {2}".format(i, day_count - 1, len(edges_for_time_interval)))
        break
    network.add_edges(edges_for_time_interval)
    i += number_of_new_days_in_interval
