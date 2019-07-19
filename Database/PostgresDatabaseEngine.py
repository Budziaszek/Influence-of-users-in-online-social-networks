import psycopg2
from Database.DatabaseEngine import DatabaseEngine


class PostgresDatabaseEngine(DatabaseEngine):
    cur = None

    def connect(self, connection_parameters):
        try:
            self.db = psycopg2.connect(str(connection_parameters))
            print("Connected to database")
        except psycopg2.OperationalError:
            print("Unable to connect to the database")
            return
        self.cur = self.db.cursor()

    def execute(self, query, parameters=None):
        if self.cur is not None:
            self.cur.execute(query, parameters)
            return self.cur.fetchall()
