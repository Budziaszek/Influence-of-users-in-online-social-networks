from enum import Enum


class MetricsType:
    value = None
    complex_description = None
    is_complex = False

    NEIGHBORS_COUNT = "neighbors_count"
    CONNECTIONS_COUNT = "connections_count"
    DENSITY = "density"
    RECIPROCITY = "reciprocity"
    JACCARD_INDEX = "jaccard_index"
    PART_IN_UNION = "part_of_neighborhood"
    PART_IN_STATIC = "part_in_static"
    CUSTOM_COMPLEX = "custom_complex"

    def __init__(self, value, complex_description=None):
        self.complex_description = complex_description
        self.is_complex = False
        if complex_description is not None:
            self.is_complex = True
        self.value = value

    def calculate(self, connection_type, graphs, graph_id, user_id, manager=None):
        if self.is_complex:
            return self._calculate_complex_type(connection_type, graphs, graph_id, user_id, manager)
        return self._calculate_basic_type(connection_type, graphs, graph_id, user_id)

    def _calculate_complex_type(self, connection_type, graphs, graph_id, user_id, manager):
        # (Basic_type_1, operator, array_creator_function)
        if self.complex_description is None:
            raise Exception("Complex value without description")
        m_1 = self.complex_description.parent_metrics_1. \
            calculate(connection_type, graphs, graph_id, user_id, manager)
        m_2 = self.complex_description.parent_metrics_2. \
            calculate(connection_type, graphs,
                      self.complex_description.get_graph_id_for_parent_metric_2(graph_id), user_id, manager)
        if self.complex_description.operator is "/":
            if m_2 != 0:
                return m_1 / m_2
            return 0
        elif self.complex_description.operator is "-":
            return m_1 - m_2
        else:
            raise Exception("Unknown operator")

    def _calculate_basic_type(self, connection_type, graphs, graph_id, user_id):
        if self.value is self.NEIGHBORS_COUNT:
            return connection_type.neighbors_count(graphs[graph_id], user_id)
        if self.value is self.CONNECTIONS_COUNT:
            return connection_type.connections_count(graphs[graph_id], user_id)
        if self.value is self.DENSITY:
            return connection_type.density(graphs[graph_id], user_id)
        if self.value is self.RECIPROCITY:
            dictionary = graphs[graph_id].reciprocity([user_id])
            return dictionary[user_id] if user_id in dictionary else 0
        if self.value is self.JACCARD_INDEX:
            return connection_type.jaccard_index(graphs[graph_id], user_id)
        if self.value is self.PART_IN_STATIC:
            return connection_type.part_of_neighborhood(graphs[graph_id], user_id)
        else:
            raise Exception("CUSTOM_COMPLEX without definition")
