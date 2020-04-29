import logging
import statistics
import sys
import _thread
import threading
from concurrent.futures.thread import ThreadPoolExecutor
import matplotlib.pyplot as plt
from time import sleep

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.uic.properties import QtCore

from Manager import Manager
from Metrics.Metrics import Metrics
from Metrics.config import neighborhood_quality_in, neighborhood_quality_out, statistics_functions, \
    clustering_scenario_3, degree_in_static, functions, clustering_scenario_1, clustering_scenario_2
from Network.GraphConnectionType import GraphConnectionType
from Network.GraphIterator import GraphIterator
from Network.NeighborhoodMode import NeighborhoodMode
from Utility.Functions import without_nan, max_mode


class ManagerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Influence of users in online social networks'
        self.neighborhood_mode = NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS
        self.left = 10
        self.top = 10
        self.width = 800
        self.height = 400
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

        button_add_metrics = QPushButton("Add metrics")
        vbox_metrics.addWidget(button_add_metrics)
        button_add_metrics.clicked.connect(self.add_metrics)

        self.create_button("Calculate", self.calculate)
        self.create_button("Display", self.display)
        self.create_button("Histogram", self.histogram)
        self.create_button("Line histogram", self.distribution_line)
        self.create_button("Statistics", self.statistics)
        self.create_button("Correlation", self.correlation)
        self.create_button("Ranking", self.ranking)
        self.create_button("Table", self.table)
        self.create_button("K-means", self.k_means)

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

    def get_metrics_definitions(self):
        metrics_definitions = []
        for i in range(self.metrics_list.count()):
            metrics_definitions.append(self.get_single_metrics_definition(self.metrics_list.item(i)))
        return metrics_definitions

    @staticmethod
    def get_single_metrics_definition(item):
        metrics, connections, iterator = item.split(' ') if isinstance(item, str) else item.text().split(' ')
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
                                     save_to_file=False,
                                     metrics=value,
                                     save_to_database=True,
                                     data_condition_function=without_nan,
                                     do_users_selection=True if value in [neighborhood_quality_in,
                                                                          neighborhood_quality_out] else False,
                                     log_fun=self.log_output.append,
                                     safe_save=False
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
                self.wait_for_graphs()
                for value in self.get_metrics_definitions():
                    self.log_output.append('Histogram for ' + value.get_name() + ".")
                    self.manager.histogram(neighborhood_mode=NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS,
                                           metrics=value,
                                           n_bins=float(n_bins) if n_bins != '' else -1,
                                           # cut=[float("-inf"), float("inf")],
                                           cut=[float(range_start) if range_start != '' else float("-inf"),
                                                float(range_end) if range_end != '' else float("inf")],
                                           half_open=False,
                                           integers=False,
                                           step=float(step) if step != '' else -1,
                                           normalize=False
                                           )
                Manager.show_plots()
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
                self.wait_for_graphs()
                for value in self.get_metrics_definitions():
                    self.log_output.append('Line histogram for ' + value.get_name() + ".")
                    self.manager.distribution_linear(
                        neighborhood_mode=NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS,
                        metrics=[value],
                        cut=(float(range_start) if range_start != '' else float("-inf"),
                             float(range_end) if range_end != '' else float("inf")),
                        n_bins=float(n_bins) if n_bins != '' else -1)
                self.manager.distribution_linear(
                    neighborhood_mode=NeighborhoodMode.COMMENTS_TO_POSTS_FROM_OTHERS,
                    metrics=self.get_metrics_definitions(),
                    cut=(float('-inf'), float('inf')),
                    n_bins=float(n_bins) if n_bins != '' else -1)
                Manager.show_plots()

        except Exception as e:
            self.log_output.append('Exception: ' + str(e))

    def statistics(self):
        try:
            self.wait_for_graphs()
            for value in self.get_metrics_definitions():
                self.log_output.append('Statistics for ' + value.get_name() + " (saved in output/statistics).")
                self.manager.statistics(
                    neighborhood_mode=self.neighborhood_mode,
                    metrics=value,
                    statistics=statistics_functions,
                    normalize=True,
                    log_fun=self.log_output.append,
                )
        except Exception as e:
            self.log_output.append('Exception: ' + str(e))

    def correlation(self):
        try:
            self.wait_for_graphs()
            values = self.get_metrics_definitions()
            self.log_output.append('Correlation for ' + str([value.get_name() for value in values])
                                   + " (saved in output/correlation).")
            self.manager.correlation(self.neighborhood_mode, values, functions)
        except Exception as e:
            self.log_output.append('Exception: ' + str(e))

    def ranking(self):
        try:
            values = self.get_metrics_definitions()
            self.log_output.append('Ranking for ' + str([value.get_name() for value in values])
                                   + " (saved in output/table).")
            self.executor.submit(self.manager.table,
                                 neighborhood_mode=self.neighborhood_mode,
                                 metrics=values,
                                 functions=functions,
                                 table_mode="index")
        except Exception as e:
            self.log_output.append('Exception: ' + str(e))

    def table(self):
        try:
            values = self.get_metrics_definitions()
            self.log_output.append('Ranking for ' + str([value.get_name() for value in values])
                                   + " (saved in output/table).")
            self.executor.submit(self.manager.table,
                                 neighborhood_mode=self.neighborhood_mode,
                                 metrics=values,
                                 functions=functions,
                                 table_mode="value")
        except Exception as e:
            self.log_output.append('Exception: ' + str(e))

    def k_means(self):
        try:
            scenarios = [clustering_scenario_1,
                         clustering_scenario_2,
                         clustering_scenario_3]
            self.options_window = OptionsWindow(parent=self,
                                                text='K-means options',
                                                range_start=True,
                                                range_end=True,
                                                n_bins=True,
                                                metrics=True,
                                                scenario=True,
                                                scenario_sel=[i + 1 for i in range(len(scenarios))],
                                                metrics_sel=[self.metrics_list.item(i).text() for i in
                                                             range(self.metrics_list.count())])
            if self.options_window.exec_():
                range_start, range_end, n_bins, step, metrics, scenario = self.options_window.get_values()
                if metrics != '':
                    items_list = self.metrics_list.findItems(metrics, Qt.MatchContains)
                    self.metrics_list.takeItem(self.metrics_list.row(items_list[0]))
                values = self.get_metrics_definitions()
                self.log_output.append('K-means clustering for ' + str(
                    [value.get_name() for value in values] if scenario == '' else str(scenario))
                                       + " (result will be saved in output/clustering).")
                if scenario == '':
                    values = [(self.neighborhood_mode, m, 1, max_mode) for m in values]
                    scenario = None
                else:
                    scenario = scenarios[int(scenario) - 1]

                self.executor.submit(self.manager.k_means,
                                     n_clusters=float(n_bins) if n_bins != '' else 6,
                                     parameters=values if scenario is None else scenario,
                                     users_selection=self.manager.select_users(
                                         neighborhood_mode=self.neighborhood_mode,
                                         metrics=ManagerApp.get_single_metrics_definition(metrics),
                                         values_start=float(range_start) if range_start != '' else float("-inf"),
                                         values_stop=float(range_end) if range_end != '' else float("inf"))
                                     if metrics != '' else None,
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
                self.wait_for_graphs()
                metrics = self.get_single_metrics_definition(metrics) if metrics != '' else None
                if metrics:
                    self.log_output.append('Display ' + metrics.get_name() + ".")
                    self.manager.display_between_range(
                        neighborhood_mode=self.neighborhood_mode,
                        metrics=metrics,
                        minimum=float(range_start) if range_start != '' else float("-inf"),
                        maximum=float(range_end) if range_end != '' else float("inf"),
                        log_fun=self.log_output.append)
        except Exception as e:
            self.log_output.append('Exception: ' + str(e))

    def closeEvent(self, event):
        self.window_metrics_selection.close()
        event.accept()


class WindowMetricsSelection(QWidget):

    def __init__(self, text, parent_method):
        QWidget.__init__(self)
        self.setWindowTitle(text)
        self.parent_method = parent_method

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
        self.parent_method(metrics + " " + connection + " " + iterator)


class OptionsWindow(QDialog):
    def __init__(self, parent, text="Options window", range_start=False, range_end=False, n_bins=False, step=False,
                 metrics=False, metrics_sel=None, scenario=False, scenario_sel=None):
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
