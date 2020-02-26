class GraphIterator:
    static_graph = None
    dynamic_graphs = None
    _graph_mode = None

    class GraphMode:
        STATIC = "static"
        DYNAMIC = "dynamic"
        DYNAMIC_CURR_NEXT = "dynamic_curr_next"
        DYNAMIC_CURR_NEXT_AND_STATIC = "dynamic_curr_next_and_static"
        ALL = "all"

    @staticmethod
    def set_graphs(static_graph, dynamic_graphs):
        GraphIterator.static_graph = static_graph
        GraphIterator.dynamic_graphs = dynamic_graphs

    def __init__(self, graph_mode):
        self.graph_mode = graph_mode
        self._graph_mode = graph_mode
        self.current_id = 0
        self.stop = False

    def get_mode(self):
        return self._graph_mode

    def next(self):
        if GraphIterator.static_graph is None or GraphIterator.dynamic_graphs is None:
            raise Exception("GraphIterator without graphs set.")
        if self.graph_mode is self.GraphMode.DYNAMIC:
            graph = GraphIterator.dynamic_graphs[self.current_id]
            if self.current_id + 1 >= len(GraphIterator.dynamic_graphs):
                self.stop = True
            else:
                self.current_id += 1
            return graph
        if self.graph_mode is self.GraphMode.STATIC:
            self.stop = True
            return GraphIterator.static_graph
        if self.graph_mode is self.GraphMode.ALL:
            self.graph_mode = self.GraphMode.DYNAMIC
            return GraphIterator.static_graph
        if self.graph_mode is self.GraphMode.DYNAMIC_CURR_NEXT \
                or self.graph_mode is self.GraphMode.DYNAMIC_CURR_NEXT_AND_STATIC:
            graph = [GraphIterator.dynamic_graphs[self.current_id], GraphIterator.dynamic_graphs[self.current_id+1]]
            if self.current_id + 2 >= len(GraphIterator.dynamic_graphs):
                self.stop = True
            else:
                self.current_id += 1
            if self.graph_mode is self.GraphMode.DYNAMIC_CURR_NEXT_AND_STATIC:
                graph.append(GraphIterator.static_graph)
            return graph

    def reset(self):
        self.stop = False
        self.current_id = 0
        self.graph_mode = self._graph_mode