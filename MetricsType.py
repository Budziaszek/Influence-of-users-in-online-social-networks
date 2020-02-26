from GraphIterator import GraphIterator


class MetricsType:
    value = None
    complex_description = None
    is_complex = False

    NEIGHBORS_COUNT = "neighbors_count"
    CONNECTIONS_COUNT = "connections_count"
    DENSITY = "density"
    RECIPROCITY = "reciprocity"
    JACCARD_INDEX_NEIGHBORS = "jaccard_index"
    PART_IN_UNION = "part_of_neighborhood"
    COMPOSITION_NEIGHBORS = "composition_neighbors"
    COMPOSITION_NEIGHBORS_PERCENTS = "composition_neighbors_percents"
    NEIGHBORS_COUNT_DIFFERENCE = "neighbors_count_difference"
    NEW_NEIGHBORS = "new_neighbors"

    # PART_IN_STATIC = "part_in_static"
    # CUSTOM_COMPLEX = "custom_complex"

    def get_name(self):
        v = self.value + "_" + self.graph_iterator.graph_mode
        return v + "_" + str(self.data) if self.data is not None else v

    def __init__(self, value, connection_type, graph_iterator=GraphIterator(GraphIterator.GraphMode.ALL), data=None):
        self.connection_type = connection_type
        self.graph_iterator = graph_iterator
        self.is_complex = False
        self.value = value
        self.data = data
        if not isinstance(connection_type, list) and self.value is self.JACCARD_INDEX_NEIGHBORS:
            self.connection_type = [connection_type, connection_type]

    def calculate(self, user_id, first_activity_date=None):
        data = []
        self.graph_iterator.reset()
        while not self.graph_iterator.stop:
            graph = self.graph_iterator.next()
            value = -1
            if first_activity_date is None or first_activity_date <= graph.end_day:
                if not isinstance(graph, list) and self.value is self.JACCARD_INDEX_NEIGHBORS:
                    graph = [graph, graph]
                value = self._calculate_basic_type(self.connection_type, graph, user_id)
            data.append(value)
        return data

    def _calculate_basic_type(self, connection_type, graph, user_id):
        if self.value is self.NEIGHBORS_COUNT:
            return connection_type.neighbors_count(graph, user_id)
        if self.value is self.CONNECTIONS_COUNT:
            return connection_type.connections_count(graph, user_id)
        if self.value is self.DENSITY:
            return connection_type.density(graph, user_id)
        if self.value is self.RECIPROCITY:
            dictionary = graph.reciprocity([user_id])
            return dictionary[user_id] if user_id in dictionary else 0
        if self.value is self.JACCARD_INDEX_NEIGHBORS:
            return self._jaccard_index(connection_type, graph, user_id)
        if self.value is self.COMPOSITION_NEIGHBORS:
            return self._neighborhood_composition(connection_type, graph, user_id, self.data)
        if self.value is self.COMPOSITION_NEIGHBORS_PERCENTS:
            return self._neighborhood_composition(connection_type, graph, user_id, self.data, percent=True)
        if self.value is self.NEIGHBORS_COUNT_DIFFERENCE:
            return self._count_difference(connection_type, graph, user_id)
        if self.value is self.NEIGHBORS_COUNT_DIFFERENCE:
            return self._count_difference(connection_type, graph, user_id)
        if self.value is self.NEW_NEIGHBORS:
            return self._new_neighbors(connection_type, graph, user_id)
        else:
            raise Exception("No metrics definition")

    @staticmethod
    def _count_difference(connection_type, graph, node):
        neighbors_1 = connection_type.neighbors(graph[0], node)
        neighbors_2 = connection_type.neighbors(graph[1], node)

        return (len(neighbors_2) - len(neighbors_1)) / (len(neighbors_2) + 1)

    @staticmethod
    def _jaccard_index(connection_type, graph, node):
        neighbors_1 = connection_type[0].neighbors(graph[0], node)
        neighbors_2 = connection_type[1].neighbors(graph[1], node)
        if len(neighbors_1) == 0 or len(neighbors_2) == 0:
            return 0
        intersection = list(set(neighbors_1).intersection(set(neighbors_2)))
        return len(intersection)/(len(neighbors_1) + len(neighbors_2) - len(intersection))

    @staticmethod
    def _new_neighbors(connection_type, graph, node):
        neighbors_1 = connection_type.neighbors(graph[0], node)
        neighbors_2 = connection_type.neighbors(graph[1], node)
        if len(neighbors_1) == 0 or len(neighbors_2) == 0:
            return 0
        difference = list(set(neighbors_2).difference(set(neighbors_1)))
        union = list(set(neighbors_1).union(set(neighbors_2)))
        return len(difference)/len(union)

    @staticmethod
    def _neighborhood_composition(connection_type, graph, node, size, percent=False):
        neighbors = connection_type.neighbors(graph, node)
        count = 0
        for neighbor in neighbors:
            neighbors_count = connection_type.neighbors_count(graph, neighbor)
            if neighbors_count >= size[0]:
                if neighbors_count <= size[1]:
                    count += 1
        if percent is True:
            return count/len(neighbors) if len(neighbors) > 0 else 0
        return count


