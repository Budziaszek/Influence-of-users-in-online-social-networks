import logging

import psycopg2
from code.data.DatabaseEngine import DatabaseEngine
from code.utility.Functions import fun_all


class PostgresDatabaseEngine(DatabaseEngine):
    cur = None

    def connect(self, connection_parameters):
        try:
            self.db = psycopg2.connect(str(connection_parameters))
            logging.info("Connected to database")
        except psycopg2.OperationalError:
            logging.error("Unable to connect to the database")
            return

    def create_activity_date_columns(self):
        if self.cur is not None and not self.does_column_exist("authors", "first_activity_date"):
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
                        tmp_cur.execute("""UPDATE %s SET %s = '%s' WHERE id = %s"""
                                        % ("authors", "first_activity_date", date, author_id))
                    n = n + 1
                data = self.cur.fetchmany(1000)
                self.db.commit()

        if self.cur is not None and not self.does_column_exist("authors", "last_activity_date"):
            self.cur.execute("""ALTER TABLE %s ADD %s date"""
                             % ("authors", "last_activity_date"))
            self.cur.execute("""SELECT author_id, date FROM comments
                            UNION
                            SELECT author_id, date FROM posts
                            ORDER BY author_id, date DESC""")
            tmp_cur = self.db.cursor()
            data = self.cur.fetchmany(1000)
            author_id = None
            while len(data) is not 0:
                n = 0
                while n < len(data):
                    if data[n][0] != author_id:
                        author_id = data[n][0]
                        date = data[n][1]
                        tmp_cur.execute("""UPDATE %s SET %s = '%s' WHERE id = %s"""
                                        % ("authors", "last_activity_date", date, author_id))
                    n = n + 1
                data = self.db.cursor().fetchmany(1000)
                self.db.commit()

    def drop_column(self, column_name, table="authors"):
        self.db.cursor().execute("""ALTER TABLE" %s DROP %s""" % (table, column_name))
        self.db.commit()

    def get_activity_dates(self):
        return dict(self.execute("SELECT id, first_activity_date FROM authors")), \
               dict(self.execute("SELECT id, last_activity_date FROM authors"))

    @staticmethod
    def lst2pgarr(alist):
        return '{' + ','.join([str(x) for x in alist]) + '}'

    def update_array_value_column(self, column_name, author_id, value):
        if not self.does_column_exist("authors", column_name):
            try:
                self.db.cursor().execute("""ALTER TABLE %s ADD %s float[]""" % ("authors", column_name))
            except psycopg2.Error:
                self.db.rollback()
        tmp_cur = self.db.cursor()
        tmp_cur.execute("""UPDATE %s SET %s = '%s' WHERE id = %s""" %
                        ("authors", column_name, self.lst2pgarr(value), author_id))
        self.db.commit()

    def get_array_value_column_for_user(self, column_name, author_id, fun=None):
        cur =  self.db.cursor()
        cur.execute("""SELECT %s FROM authors WHERE id = %s""" % (column_name, author_id))
        self.db.commit()
        if fun == "all":
            return cur.fetchall()[0]
        if fun is not None:
            return fun(cur.fetchall()[0][0])
        return cur.fetchall()[0][0]

    def get_array_value_column(self, column_name, fun=None):
        cur = self.db.cursor()
        cur.execute("""SELECT id, %s FROM authors ORDER BY id""" % column_name)
        self.db.commit()
        if fun == fun_all:
            return dict(cur.fetchall())
        if fun is not None:
            return {x[0]: fun(x[1]) for x in cur.fetchall()}
        return {x[0]: x[1][0] for x in cur.fetchall()}

    def get_min_max_array_value_column(self, column_name, fun=None):
        cur = self.db.cursor()
        cur.execute("""SELECT %s FROM authors ORDER BY id""" % column_name)
        self.db.commit()
        data = [d[0][0] for d in cur.fetchall()]
        if fun is not None and isinstance(data[0], list):
            data = [fun(d) for d in data]
            return min(data), max(data)
        else:
            return min(data), max(data)

    def does_column_exist(self, table, column):
        self.db.cursor().execute("""SELECT count(*)
                                 FROM information_schema.columns
                                 WHERE table_name = '%s'
                                 AND column_name = '%s'"""
                         % (table, column))
        return self.db.cursor().fetchone()[0] is not 0

    def get_dates_range(self, column):
        return self.execute("SELECT min(date), max(date) FROM " + column).pop()

    def get_posts(self, day_start, day_end):
        # return self.execute("""SELECT a.id,
        #                        (SELECT count(*)
        #                         FROM posts p
        #                         WHERE p.author_id = a.id
        #                         AND p.date BETWEEN """
        #                     + "'" + str(day_start) + "' and '" + str(day_end) + "'"
        #                                                                         ") FROM authors a")
        return self.execute("""SELECT p.author_id
                                        FROM posts p 
                                        WHERE p.date BETWEEN """
                            + "'" + str(day_start) + "' and '" + str(day_end) + "'")

    def get_responses(self, day_start, day_end):
        # return self.execute("""SELECT a.id,
        #                        (SELECT count(*)
        #                         FROM comments c
        #                         WHERE c.author_id = a.id
        #                         AND c.date BETWEEN """
        #                     + "'" + str(day_start) + "' and '" + str(day_end) + "'"
        #                                                                         ") FROM authors a")
        return self.execute("""SELECT p.author_id
                                        FROM comments c
                                        INNER JOIN posts p 
                                        ON c.post_id = p.id 
                                        WHERE c.date BETWEEN """
                            + "'" + str(day_start) + "' and '" + str(day_end) + "'"
                            + "AND c.author_id=p.author_id")

    def get_responses_to_posts(self, day_start, day_end, include_responses_from_author):
        return self.execute("""SELECT c.author_id, p.author_id 
                                FROM comments c
                                INNER JOIN posts p 
                                ON c.post_id = p.id 
                                WHERE c.date BETWEEN """
                            + "'" + str(day_start) + "' and '" + str(day_end) + "'"
                            + ("" if include_responses_from_author
                               else "AND c.author_id!=p.author_id"))

    def get_responses_to_comments(self, day_start, day_end, include_responses_from_author):
        return self.execute("""SELECT c.author_id, p.author_id
                                FROM comments c
                                INNER JOIN comments p 
                                ON c.parentcomment_id = p.id
                                WHERE c.parentcomment_id IS NOT NULL
                                AND c.date BETWEEN """
                            + "'" + str(day_start) + "' and '" + str(day_end) + "'"
                            + ("" if include_responses_from_author
                               else "AND c.author_id!=p.author_id"))

    def get_interesting_users(self, limit=50):
        return self.execute("""SELECT id 
                                FROM authors 
                                ORDER BY po_in_degree_static DESC
                                LIMIT """ + str(limit))

    def get_authors_parameter(self, parameter):
        return dict(self.execute("SELECT id, " + parameter + " FROM authors"))

    def execute(self, query, parameters=None):
        cur = self.db.cursor()
        cur.execute(query, parameters)
        return cur.fetchall()
