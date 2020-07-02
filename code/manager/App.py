import logging
import matplotlib
import statistics
import threading
from concurrent.futures.thread import ThreadPoolExecutor
from time import sleep

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from code.data.ResultsDisplay import data_statistics, distribution_linear, show_plots, histogram, \
    category_histogram
from code.manager.Manager import Manager
from code.metrics.Metrics import Metrics
from code.metrics.config import clustering_scenario_3, functions, clustering_scenario_1, clustering_scenario_2
from code.network.GraphConnectionType import GraphConnectionType
from code.network.GraphIterator import GraphIterator
from code.network.NeighborhoodMode import NeighborhoodMode
from code.utility.Functions import without_nan, max_mode, fun_all

# import matplotlib
matplotlib.use('TkAgg')


class ManagerApp(QWidget):
    """Warning! This GUI is intended for testing purposes. Improper use may result in an error"""
    def __init__(self):
        super().__init__()
        self.title = 'Influence of users in online social networks'
        self.neighborhood_mode = NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS
        self.left = 10
        self.top = 10
        self.width = 800
        self.height = 400
        self.users_selection = None
        self.metrics_list = QListWidget()
        self.metrics_list_unactivated = QListWidget()
        self.log_output = QTextEdit()
        self.window_metrics_selection = None
        self.options_window = None
        self.vbox_actions = QVBoxLayout()
        self.vbox_actions.setAlignment(Qt.AlignTop)
        self.executor = ThreadPoolExecutor(1)
        self.initUI()
        self.manager = Manager(connection_parameters="dbname='salon24' "
                                                     "user='sna_user' "
                                                     "host='localhost' "
                                                     "password='sna_password'",
                               test=False)
        self.log_output.append('Waiting for graphs to be created...')
        self.load_t = threading.Thread(target=self.manager.check_graphs,
                                       kwargs={'neighborhood_mode': self.neighborhood_mode})
        self.load_t.start()
        self.category_data = self.manager.get_category(self.users_selection)
        threading.Thread(target=self.check_graphs).start()

        # self.manager.plot_overlapping_ranges()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        vbox = QHBoxLayout()
        hbox_main = QHBoxLayout()

        vbox_metrics = QVBoxLayout()

        vbox_metrics.addWidget(self.metrics_list)
        self.metrics_list.clicked.connect(self.clicked_activated)

        vbox_metrics.addWidget(self.metrics_list_unactivated)
        self.metrics_list_unactivated.clicked.connect(self.clicked_unactivated)

        button_add_metrics = QPushButton("Add Metrics")
        vbox_metrics.addWidget(button_add_metrics)
        button_add_metrics.clicked.connect(self.add_metrics)

        self.create_button("Calculate", self.calculate)
        self.create_button("Display", self.display)
        self.create_button("Histogram", self.histogram)
        self.create_button("Category Histogram", self.category_histogram)
        self.create_button("Line histogram", self.distribution_line)
        self.create_button("Statistics", self.statistics)
        self.create_button("Correlation", self.correlation)
        self.create_button("Ranking", self.ranking)
        self.create_button("Table", self.table)
        self.create_button("Agglomerative Clustering", self.agglomerative_clustering)
        self.create_button("K-means", self.k_means)
        self.create_button("Prediction", self.prediction)
        self.create_button("Selection", self.selection)
        self.create_button("Clear selection", self.clear_selection)

        hbox_main.addLayout(vbox_metrics)
        hbox_main.addLayout(self.vbox_actions)

        self.log_output.setReadOnly(True)
        self.log_output.setLineWrapMode(QTextEdit.NoWrap)
        font = self.log_output.font()
        font.setFamily("Courier")
        font.setPointSize(10)
        vbox.addLayout(hbox_main)
        vbox.addWidget(self.log_output)

        self.setLayout(vbox)
        self.show()

    def create_button(self, name, fun):
        button = QPushButton(name)
        button.clicked.connect(fun)
        self.vbox_actions.addWidget(button)
        return button

    def clicked_activated(self):
        item = self.metrics_list.takeItem(self.metrics_list.currentRow())
        self.metrics_list_unactivated.addItem(item)
        logging.info("Deactivated " + item.text())

    def clicked_unactivated(self):
        item = self.metrics_list_unactivated.takeItem(self.metrics_list_unactivated.currentRow())
        self.metrics_list.addItem(item)
        logging.info("Activated " + item.text())

    def add_metrics(self):
        self.window_metrics_selection = WindowMetricsSelection("Metrics selection", self.submit_metrics)
        self.window_metrics_selection.show()

    def submit_metrics(self, item):
        logging.info("Added " + item)
        self.metrics_list.addItem(item)

    def get_metrics_definitions(self, stats=False):
        metrics_definitions = []
        for i in range(self.metrics_list.count()):
            metrics_definitions.append(self.get_single_metrics_definition(self.metrics_list.item(i), stats))
        return metrics_definitions

    @staticmethod
    def get_stats_fun(s):
        if s == 'len':
            return len
        elif s == 'mean':
            return statistics.mean
        elif s == 'mode':
            return max_mode
        elif s == 'max':
            return max
        elif s == 'min':
            return min
        elif s == 'median':
            return statistics.median
        elif s == 'all':
            return fun_all

    @staticmethod
    def get_single_metrics_definition(item, stats=False):
        x = item.split(' ') if isinstance(item, str) else item.text().split(' ')
        metrics, connections, iterator = x[0], x[1], x[2]
        if stats:
            if len(x) == 4:
                stats = ManagerApp.get_stats_fun(x[3])
                return Metrics(metrics, GraphConnectionType(connections), GraphIterator(iterator)), stats
            else:
                return Metrics(metrics, GraphConnectionType(connections), GraphIterator(iterator)), None
        return Metrics(metrics, GraphConnectionType(connections), GraphIterator(iterator))

    def check_graphs(self):
        self.wait_for_graphs()
        self.log_output.append('Graphs created.')

    def wait_for_graphs(self):
        while self.load_t.is_alive():
            sleep(1.0)

    def calculate(self):
        try:
            self.wait_for_graphs()
            for value in self.get_metrics_definitions():
                self.log_output.append('Calculating ' + value.get_name() + "...")
                self.executor.submit(fn=self.manager.calculate,
                                     metrics=value,
                                     condition_fun=without_nan,
                                     log_fun=self.log_output.append,
                                     )

        except Exception as e:
            self.log_output.append('Exception: ' + str(e))

    def histogram(self):
        try:
            self.options_window = OptionsWindow(parent=self, text='Histogram options',
                                                range_start=True,
                                                range_end=True,
                                                n_bins=True,
                                                step=True)
            if self.options_window.exec_():
                range_start, range_end, n_bins, step, metrics, scenario = self.options_window.get_values()
                for value in self.get_metrics_definitions():
                    self.log_output.append('Histogram for ' + value.get_name() + ".")
                    histogram(title=value.get_name(),
                              data=self.manager.get_data(
                                  neighborhood_mode=NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS,
                                  metrics=value,
                                  users_selection=self.users_selection,
                                  cut_down=float(range_start) if range_start != '' else float("-inf"),
                                  cut_up=float(range_end) if range_end != '' else float("inf")),
                              n_bins=int(n_bins) if n_bins != '' else 10,
                              half_open=False,
                              integers=False,
                              step=float(step) if step != '' else -1,
                              normalize=False
                              )
                show_plots()
        except Exception as e:
            self.log_output.append('Exception: ' + str(e))

    def category_histogram(self):
        try:
            self.options_window = OptionsWindow(parent=self, text='Histogram options',
                                                range_start=True,
                                                range_end=True,
                                                n_bins=True,
                                                step=True)
            if self.options_window.exec_():
                range_start, range_end, n_bins, step, metrics, scenario = self.options_window.get_values()
                for value in self.get_metrics_definitions(True):
                    name = value[0].get_name() + ("_" + value[1].__name__ if value[1] is not None else '')
                    self.log_output.append('Histogram for ' + name + ".")
                    category_histogram(title=name,
                                       category_data=self.category_data,
                                       labels=["0", "1-10", "11-100", "101-1000", ">1000"],
                                       data=self.manager.get_data(
                                           neighborhood_mode=NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS,
                                           metrics=value[0],
                                           fun=value[1],
                                           users_selection=self.users_selection,
                                           cut_down=float(range_start) if range_start != '' else float("-inf"),
                                           cut_up=float(range_end) if range_end != '' else float("inf")),
                                       n_bins=int(n_bins) if n_bins != '' else 10,
                                       )
                show_plots()
        except Exception as e:
            self.log_output.append('Exception: ' + str(e))

    def distribution_line(self):
        try:
            self.options_window = OptionsWindow(parent=self, text='Line histogram options',
                                                range_start=True,
                                                range_end=True,
                                                n_bins=True)
            if self.options_window.exec_():
                range_start, range_end, n_bins, step, metrics, scenario = self.options_window.get_values()
                values = self.get_metrics_definitions()
                data = {}
                for value in values:
                    self.log_output.append('Line histogram for ' + value.get_name() + ".")
                    d = self.manager.get_data(neighborhood_mode=self.neighborhood_mode,
                                              metrics=value,
                                              cut_down=float(range_start) if range_start != '' else float("-inf"),
                                              cut_up=float(range_end) if range_end != '' else float("inf"))
                    data[value.get_name()] = d
                    # distribution_linear(data={value.get_name(): d},
                    #                     n_bins=float(n_bins) if n_bins != '' else -1)
                distribution_linear(data=data,
                                    n_bins=float(n_bins) if n_bins != '' else -1)
                show_plots()
        except Exception as e:
            self.log_output.append('Exception: ' + str(e))

    def statistics(self):
        try:
            # scenarios = ['None', 'in_degree>0, out_degree=0']
            # self.options_window = OptionsWindow(parent=self, text='Line histogram options',
            #                                     scenario=True,
            #                                     scenario_sel=scenarios)
            # if self.options_window.exec_():
            #     range_start, range_end, n_bins, step, Metrics, scenario = self.options_window.get_values()
            #     if scenario == scenarios[1]:
            #         users = self.manager.select_users(
            #             neighborhood_mode=self.neighborhood_mode,
            #             Metrics=ManagerApp.get_single_metrics_definition(Metrics),
            #             values_start=float(range_start) if range_start != '' else float("-inf"),
            #             values_stop=float(range_end) if range_end != '' else float("inf"))
            #
            for value in self.get_metrics_definitions(True):
                self.log_output.append('Statistics for ' + value[0].get_name() + " (saved in output/data_statistics).")
                data_statistics(
                    title=value[0].get_name() + ("_" + value[1].__name__ if value[1] is not None else ''),
                    data=self.manager.get_data(neighborhood_mode=self.neighborhood_mode,
                                               metrics=value[0],
                                               fun=value[1],
                                               users_selection=self.users_selection),
                    normalize=False,
                    log_fun=self.log_output.append,
                )
        except Exception as e:
            self.log_output.append('Exception: ' + str(e))

    def correlation(self):
        try:
            values = self.get_metrics_definitions()
            self.log_output.append('Correlation for ' + str([value.get_name() for value in values])
                                   + " (saved in output/correlation).")
            self.manager.correlation(self.neighborhood_mode, values, functions, log_fun=self.log_output.append)
        except Exception as e:
            self.log_output.append('Exception: ' + str(e))

    def ranking(self):
        self.all_to_file('index')

    def table(self):
        self.all_to_file('value')

    def all_to_file(self, mode):
        try:
            values = self.get_metrics_definitions(True)
            self.executor.submit(self.manager.table,
                                 neighborhood_mode=self.neighborhood_mode,
                                 metrics=[v[0] for v in values],
                                 functions=[v[1] for v in values],
                                 table_mode=mode,
                                 log_fun=self.log_output.append)
        except Exception as e:
            self.log_output.append('Exception: ' + str(e))

    def k_means(self):
        self.clustering("k-means")
        pass

    def agglomerative_clustering(self):
        self.clustering('agglomerative')

    def clustering(self, t):
        try:
            scenarios = [clustering_scenario_1,
                         clustering_scenario_2,
                         clustering_scenario_3]
            self.options_window = OptionsWindow(parent=self,
                                                text='Clustering options',
                                                n_bins=True,
                                                scenario=True,
                                                scenario_sel=[i + 1 for i in range(len(scenarios))])
            if self.options_window.exec_():
                range_start, range_end, n_bins, step, metrics, scenario = self.options_window.get_values()
                values = self.get_metrics_definitions(True)
                self.log_output.append('Clustering for ' + str(
                    [value[0].get_name() + ("_" + value[1].__name__ if value[1] is not None else '')
                     for value in values] if scenario == '' else str(scenario))
                                       + " (result will be saved in output/clustering).")
                if scenario == '':
                    values = [(self.neighborhood_mode, v[0], 1, v[1]) for v in values]
                    scenario = None
                else:
                    scenario = scenarios[int(scenario) - 1]
                fun = self.manager.k_means
                if t == 'agglomerative':
                    fun = self.manager.agglomerative_clustering
                self.executor.submit(fun,
                                     n_clusters=int(n_bins) if n_bins != '' else 6,
                                     parameters=values if scenario is None else scenario,
                                     users_selection=self.users_selection,
                                     log_fun=self.log_output.append,
                                     )
        except Exception as e:
            self.log_output.append('Exception: ' + str(e))

    def prediction(self):
        try:
            values = self.get_metrics_definitions(True)
            self.log_output.append('Prediction for ' + str(
                [value[0].get_name() + ("_" + value[1].__name__ if value[1] is not None else '')
                 for value in values]) + " (result will be saved in output/prediction).")
            values = [(self.neighborhood_mode, v[0], 1, v[1]) for v in values]
            self.executor.submit(self.manager.prediction,
                                 parameters=values,
                                 users_selection=self.users_selection,
                                 log_fun=self.log_output.append,
                                 )
        except Exception as e:
            self.log_output.append('Exception: ' + str(e))

    def display(self):
        try:
            self.options_window = OptionsWindow(parent=self, text='Display options',
                                                range_start=True,
                                                range_end=True,
                                                metrics=True,
                                                metrics_sel=[self.metrics_list.item(i).text() for i in
                                                             range(self.metrics_list.count())])
            if self.options_window.exec_():
                range_start, range_end, n_bins, step, metrics, scenario = self.options_window.get_values()
                metrics = self.get_single_metrics_definition(metrics, True) if metrics != '' else None
                if metrics:
                    s = ' (' + str(metrics[1].__name__) + ')' if metrics[1] is not None else ''
                    self.log_output.append('Display ' + metrics[0].get_name() + s + ".")
                    self.manager.display_between_range(
                        neighborhood_mode=self.neighborhood_mode,
                        metrics=metrics[0],
                        minimum=float(range_start) if range_start != '' else float("-inf"),
                        maximum=float(range_end) if range_end != '' else float("inf"),
                        stats_fun=metrics[1],
                        log_fun=self.log_output.append)
        except Exception as e:
            self.log_output.append('Exception: ' + str(e))

    def selection(self):
        try:
            self.options_window = OptionsWindow(parent=self, text='Selection options',
                                                range_start=True,
                                                range_end=True,
                                                metrics=True,
                                                metrics_sel=[self.metrics_list.item(i).text() for i in
                                                             range(self.metrics_list.count())])
            if self.options_window.exec_():
                range_start, range_end, n_bins, step, metrics, scenario = self.options_window.get_values()
                if metrics:
                    self.users_selection = \
                        self.manager.select_users(
                            selection=self.users_selection,
                            neighborhood_mode=self.neighborhood_mode,
                            metrics=self.get_single_metrics_definition(metrics),
                            values_start=float(range_start) if range_start != '' else float("-inf"),
                            values_stop=float(range_end) if range_end != '' else float("inf"))
                    self.log_output.append("Selected.")
        except Exception as e:
            self.log_output.append('Exception: ' + str(e))

    def clear_selection(self):
        self.users_selection = None

    def stability(self):
        self.options_window = OptionsWindow(parent=self, text='Display options',
                                            range_start=True,
                                            range_end=True,
                                            metrics=True,
                                            metrics_sel=[self.metrics_list.item(i).text() for i in
                                                         range(self.metrics_list.count())])
        if self.options_window.exec_():
            range_start, range_end, n_bins, step, metrics, scenario = self.options_window.get_values()
            try:
                if metrics != '':
                    items_list = self.metrics_list.findItems(metrics, Qt.MatchContains)
                    self.metrics_list.takeItem(self.metrics_list.row(items_list[0]))
                self.log_output.append('Stability...')
                values = self.get_metrics_definitions()
                users = None
                if metrics != '':
                    users = self.manager.select_users(
                        neighborhood_mode=self.neighborhood_mode,
                        metrics=ManagerApp.get_single_metrics_definition(metrics),
                        values_start=float(range_start) if range_start != '' else float("-inf"),
                        values_stop=float(range_end) if range_end != '' else float("inf"))
                for value in values:
                    self.log_output.append('Stability for ' + value.get_name()
                                           + " (result will be saved in output/stability).")
                    self.executor.submit(self.manager.stability,
                                         neighborhood_mode=self.neighborhood_mode,
                                         metrics=value,
                                         users_selection=users
                                         )
            except Exception as e:
                self.log_output.append('Exception: ' + str(e))

    def closeEvent(self, event):
        self.window_metrics_selection.close()
        event.accept()


class WindowMetricsSelection(QWidget):

    def __init__(self, text, parent_method, dynamic_stats=True):
        QWidget.__init__(self)
        self.setWindowTitle(text)
        self.parent_method = parent_method
        self.stats = QComboBox()

        layout = QGridLayout()

        self.combobox_metrics = QComboBox()
        self.combobox_metrics.addItems(Metrics.METRICS_LIST)
        layout.addWidget(self.combobox_metrics)

        self.combobox_connection_types = QComboBox()
        self.combobox_connection_types.addItems(GraphConnectionType.CONNECTION_TYPES)
        layout.addWidget(self.combobox_connection_types)

        self.combobox_iterator_types = QComboBox()
        self.combobox_iterator_types.addItems(GraphIterator.ITERATOR.ITERATOR_TYPES)
        layout.addWidget(self.combobox_iterator_types)

        if dynamic_stats:
            self.stats.addItem('')
            self.stats.addItem('len')
            self.stats.addItem('mean')
            self.stats.addItem('mode')
            self.stats.addItem('max')
            self.stats.addItem('min')
            self.stats.addItem('median')
            self.stats.addItem('all')
            layout.addWidget(self.stats)

        self.button_add = QPushButton('Add')
        self.button_add.clicked.connect(self.add)
        layout.addWidget(self.button_add)

        self.button_close = QPushButton('Close')
        self.button_close.clicked.connect(self.close)
        layout.addWidget(self.button_close)

        self.setLayout(layout)

    def add(self):
        metrics = self.combobox_metrics.currentText()
        connection = self.combobox_connection_types.currentText()
        iterator = self.combobox_iterator_types.currentText()

        stats = ''
        if (iterator == 'dynamic' or iterator == 'dynamic_curr_next') and self.stats.currentText() != '':
            stats = ' ' + self.stats.currentText()
        self.parent_method(metrics + " " + connection + " " + iterator + stats)


class OptionsWindow(QDialog):
    def __init__(self, parent, text="Options window", range_start=False, range_end=False, n_bins=False, step=False,
                 metrics=False, metrics_sel=None, scenario=False, scenario_sel=None, dynamics_stats=False):
        super(OptionsWindow, self).__init__(parent)
        self.setWindowTitle(text)

        layout = QVBoxLayout()
        self.edit_range_start = QLineEdit()
        self.edit_range_end = QLineEdit()
        self.edit_number_of_bins = QLineEdit()
        self.edit_step = QLineEdit()
        self.metrics = QComboBox()
        self.scenario = QComboBox()

        if range_start:
            hbox_range_start = QHBoxLayout()
            label = QLabel("Range start: ")
            hbox_range_start.addWidget(label)
            self.edit_range_start.setValidator(QDoubleValidator())
            hbox_range_start.addWidget(self.edit_range_start)
            layout.addLayout(hbox_range_start)

        if range_end:
            hbox_range_end = QHBoxLayout()
            label = QLabel("Range end: ")
            hbox_range_end.addWidget(label)
            self.edit_range_end.setValidator(QDoubleValidator())
            hbox_range_end.addWidget(self.edit_range_end)
            layout.addLayout(hbox_range_end)

        if n_bins:
            hbox_number_of_bins = QHBoxLayout()
            label = QLabel("Number of bins: ")
            hbox_number_of_bins.addWidget(label)
            self.edit_number_of_bins.setValidator(QIntValidator())
            hbox_number_of_bins.addWidget(self.edit_number_of_bins)
            layout.addLayout(hbox_number_of_bins)

        if step:
            hbox_step = QHBoxLayout()
            label = QLabel("Step: ")
            hbox_step.addWidget(label)
            self.edit_step.setValidator(QDoubleValidator())
            hbox_step.addWidget(self.edit_step)
            layout.addLayout(hbox_step)

        if metrics:
            hbox_metrics = QHBoxLayout()
            label = QLabel("Metrics (for users selection): ")
            hbox_metrics.addWidget(label)
            self.metrics.addItem('')
            self.metrics.addItems(metrics_sel)
            hbox_metrics.addWidget(self.metrics)
            layout.addLayout(hbox_metrics)

        if scenario:
            hbox_scenario = QHBoxLayout()
            label = QLabel("Scenario: ")
            hbox_scenario.addWidget(label)
            self.scenario.addItem('')
            self.scenario.addItems([str(s) for s in scenario_sel])
            hbox_scenario.addWidget(self.scenario)
            layout.addLayout(hbox_scenario)

        hbox_buttons = QHBoxLayout()

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        layout.addWidget(self.buttonBox)
        layout.addLayout(hbox_buttons)

        self.setLayout(layout)

    def get_values(self):
        return self.edit_range_start.text(), \
               self.edit_range_end.text(), \
               self.edit_number_of_bins.text(), \
               self.edit_step.text(), \
               self.metrics.currentText(), \
               self.scenario.currentText()
