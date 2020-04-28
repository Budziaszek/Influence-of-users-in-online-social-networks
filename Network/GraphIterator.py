class GraphIterator:
    static_graph = None
    dynamic_graphs = None
    _graph_mode = []

    class ITERATOR:
        STATIC = "static"
        DYNAMIC = "dynamic"
        DYNAMIC_CURR_NEXT = "dynamic_curr_next"

        ITERATOR_TYPES = [
            STATIC,
            DYNAMIC,
            DYNAMIC_CURR_NEXT
        ]

    @staticmethod
    def set_graphs(static_graph, dynamic_graphs):
        GraphIterator.static_graph = static_graph
        GraphIterator.dynamic_graphs = dynamic_graphs

    def __init__(self, graph_mode):
        self.graph_mode = [graph_mode] if not isinstance(graph_mode, list) else graph_mode
        self._graph_mode = self.graph_mode
        self.current_id = 0
        self.stop = False

    def get_mode(self):
        return self._graph_mode

    def next(self):
        if GraphIterator.static_graph is None or GraphIterator.dynamic_graphs is None:
            raise Exception("GraphIterator without graphs set.")
        dynamic_count = self.graph_mode.count(GraphIterator.ITERATOR.DYNAMIC)
        dynamic_curr_next_count = self.graph_mode.count(GraphIterator.ITERATOR.DYNAMIC_CURR_NEXT)
        if dynamic_count + dynamic_curr_next_count > 1:
            raise Exception("Incorrect GraphIterator mode configuration. Only one type of dynamic mode is allowed.")
        graphs = []
        for mode in self.graph_mode:
            if mode == self.ITERATOR.DYNAMIC:
                graphs.append(GraphIterator.dynamic_graphs[self.current_id])
                self.current_id += 1
            elif mode == self.ITERATOR.STATIC:
                graphs.append(GraphIterator.static_graph)
            elif mode == self.ITERATOR.DYNAMIC_CURR_NEXT:
                graphs.extend([GraphIterator.dynamic_graphs[self.current_id],
                               GraphIterator.dynamic_graphs[self.current_id + 1]])
                self.current_id += 1
            if self.current_id >= len(GraphIterator.dynamic_graphs) \
                    or dynamic_curr_next_count > 0 and self.current_id + 1 >= len(GraphIterator.dynamic_graphs) \
                    or dynamic_count + dynamic_curr_next_count == 0:
                self.stop = True
        return graphs if len(graphs) > 1 else graphs[0]

    def reset(self):
        self.stop = False
        self.current_id = 0
        self.graph_mode = self._graph_mode


