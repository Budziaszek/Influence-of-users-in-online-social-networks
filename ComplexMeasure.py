from collections import defaultdict

from Network.GraphConnectionType import GraphConnectionType


class MeasureGraphIterator:
    static_graph = None
    dynamic_graphs = None

    class GraphMode:
        STATIC = "static"
        DYNAMIC = "dynamic"

    @staticmethod
    def set_graphs(static_graph, dynamic_graphs):
        MeasureGraphIterator.static_graph = static_graph
        MeasureGraphIterator.dynamic_graphs = dynamic_graphs

    def get_name(self):
        name = self.measure.get_name() + "_" + self.graph_mode
        if self.graph_mode is self.GraphMode.DYNAMIC:
            name += str(self.start_id)
        return name

    def __init__(self, measure, graph_mode, start_id=None, end_id=None):
        self.measure = measure
        self.graph_mode = graph_mode
        self.start_id = start_id
        self.current_id = start_id
        self.end_id = end_id
        self.stop = False

    def get_with_measure(self, measure):
        self.measure = measure
        return self

    def next(self):
        if MeasureGraphIterator.static_graph is None or MeasureGraphIterator.dynamic_graphs is None:
            raise Exception("GraphIterator without graphs set.")
        if self.graph_mode is self.GraphMode.DYNAMIC:
            end = len(self.dynamic_graphs) if self.end_id is None else len(self.dynamic_graphs) + self.end_id
            self.current_id += 1
            if self.current_id + 1 > end:
                self.stop = True
            return MeasureGraphIterator.dynamic_graphs[self.current_id - 1]
        if self.graph_mode is self.GraphMode.STATIC:
            self.stop = True
            return MeasureGraphIterator.static_graph

    def reset(self):
        if MeasureGraphIterator.static_graph is None or MeasureGraphIterator.dynamic_graphs is None:
            raise Exception("GraphIterator without graphs set.")
        self.current_id = self.start_id
        self.stop = False


class MeasureOperation:
    class SetOperation:
        UNION = "union"
        INTERSECTION = "intersection"
        DIFFERENCE = "difference"

    class SetReduction:
        LENGTH = "length"

    class ValueOperator:
        ADDITION = "+"
        SUBTRACTION = "-"
        MULTIPLICATION = "*"
        DIVISION = "/"

    def __init__(self, measure_1_name, operator, measure_2_name=None):
        self.measure_1_name = measure_1_name
        self.operator = operator
        self.measure_2_name = measure_2_name

    def get_name(self):
        name = self.measure_1_name + "_" + self.operator
        if self.measure_2_name is not None:
            name += "_" + self.measure_2_name
        return name

    def calculate(self, measure):
        value_1 = measure[self.measure_1_name]
        value_2 = None
        key = self.get_name()
        if self.measure_2_name is not None:
            value_2 = measure[self.measure_2_name]
        value = []
        if value_2 is not None:
            if self.operator is self.SetOperation.UNION:
                for v_1, v_2 in zip(value_1, value_2):
                    value.append(list(set(v_1).union(set(v_2))))
            elif self.operator is self.SetOperation.INTERSECTION:
                for v_1, v_2 in zip(value_1, value_2):
                    value.append(list(set(v_1).intersection(set(v_2))))
            elif self.operator is self.SetOperation.DIFFERENCE:
                for v_1, v_2 in zip(value_1, value_2):
                    value.append(list(set(v_1).difference(set(v_2))))
            elif self.operator is self.ValueOperator.ADDITION:
                for v_1, v_2 in zip(value_1, value_2):
                    value.append(v_1 + v_2)
            elif self.operator is self.ValueOperator.SUBTRACTION:
                for v_1, v_2 in zip(value_1, value_2):
                    value.append(v_1 - v_2)
            elif self.operator is self.ValueOperator.MULTIPLICATION:
                for v_1, v_2 in zip(value_1, value_2):
                    value.append(v_1 * v_2)
            elif self.operator is self.ValueOperator.DIVISION:
                for v_1, v_2 in zip(value_1, value_2):
                    if v_2 != 0:
                        value.append(v_1 / v_2)
                    else:
                        value.append(0)
        elif self.operator is self.SetReduction.LENGTH:
            for v_1 in value_1:
                value.append(len(v_1))

        return key, value


class BasicMeasure:
    value = None
    name = None

    NEIGHBORS = "neighbors"
    CONNECTIONS = "connections_count"
    DENSITY = "density"
    RECIPROCITY = "reciprocity"

    def get_name(self):
        return self.value + "_" + self.connection_type.value

    def __init__(self, value, connection_type):
        self.value = value
        self.connection_type = connection_type
        self.name = value + "_" + connection_type.value

    def calculate(self, graph, user_id):
        if self.value is self.NEIGHBORS:
            return self.connection_type.neighbors(graph, user_id)
        if self.value is self.CONNECTIONS:
            return self.connection_type.connections(graph, user_id)
        if self.value is self.DENSITY:
            return self.connection_type.density(graph, user_id)
        if self.value is self.RECIPROCITY:
            dictionary = graph.reciprocity([user_id])
            return dictionary[user_id] if user_id in dictionary else 0


class ComplexMeasure:
    def __init__(self, value, operators, measure_graph_iterators):
        self.measure_graph_iterators = measure_graph_iterators
        self.operators = operators
        self.value = value

    @staticmethod
    def is_author_active(first_activity_date, end_day):
        if end_day is None:
            return True
        if first_activity_date is None:
            return False
        return first_activity_date <= end_day

    def _remove_before_first_activity(self, data, dynamic_graphs, first_activity_date):
        if isinstance(data, list):
            for g_id, graph in enumerate(dynamic_graphs):
                if not self.is_author_active(first_activity_date, graph.end_day):
                    if g_id < len(data):
                        data[g_id] = ''
        return data

    def _is_graph_to_calculate(self):
        for graph_iterator in self.measure_graph_iterators:
            if not graph_iterator.stop:
                return True
        return False

    def _reset_iterators(self):
        for graph_iterator in self.measure_graph_iterators:
            graph_iterator.reset()

    def calculate(self, user_id, first_activity_date):
        data = defaultdict(list)
        # Calculate statistics_values for each measure
        self._reset_iterators()
        while self._is_graph_to_calculate():
            for i, measure_iterator in enumerate(self.measure_graph_iterators):
                graph = self.measure_graph_iterators[i].next()
                value = self.measure_graph_iterators[i].measure.calculate(graph, user_id)
                if not ComplexMeasure.is_author_active(first_activity_date, graph.end_day):
                    value = []
                data[measure_iterator.get_name()].append(value)
        value = None
        # print(statistics_values)
        for i, operator in enumerate(self.operators):
            key, value = operator.calculate(data)
            data[key] = value
            # print(key, value)
        return value
