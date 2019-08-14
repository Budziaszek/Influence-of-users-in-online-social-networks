import psycopg2
from Data.DatabaseEngine import DatabaseEngine


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

    def create_first_activity_date_column(self):
        if self.cur is not None and not self.does_column_exist("authors", "first_activity_date"):
            print("Add first_activity_date column")
            self.cur.execute("""ALTER TABLE %s ADD %s date"""
                             % ("authors", "first_activity_date"))
            self.cur.execute("""SELECT author_id, date FROM comments
                            UNION
                            SELECT author_id, date FROM posts
                            ORDER BY author_id, date""")
            tmp_cur = self.db.cursor()
            data = self.cur.fetchmany(1000)
            author_id = None
            while len(data) is not 0:
                n = 0
                while n < len(data):
                    if data[n][0] != author_id:
                        author_id = data[n][0]
                        date = data[n][1]
                        # print("author", author_id, date)
                        tmp_cur.execute("""UPDATE %s SET %s = '%s' WHERE id = %s"""
                                        % ("authors", "first_activity_date", date, author_id))
                    n = n + 1
                data = self.cur.fetchmany(1000)
                self.db.commit()

        # self.cur.execute("""UPDATE %s SET %s = (SELECT min(min_date) FROM (
        #            SELECT min(comments.date) AS min_date FROM comments WHERE author_id = %s
        #            UNION
        #            SELECT min(posts.date) AS min_date FROM posts WHERE author_id = %s
        #            ) AS X) WHERE id = %s""" % ("authors", "first_activity_date", author_id, author_id, author_id))

    def does_column_exist(self, table, column):
        if self.cur is not None:
            self.cur.execute("""SELECT count(*)
                                     FROM information_schema.columns
                                     WHERE table_name = '%s'
                                     AND column_name = '%s'"""
                             % (table, column))
            return self.cur.fetchone()[0] is not 0

    def execute(self, query, parameters=None):
        if self.cur is not None:
            self.cur.execute(query, parameters)
            return self.cur.fetchall()