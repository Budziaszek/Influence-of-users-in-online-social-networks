class GraphIterator:
    static_graph = None
    dynamic_graphs = None
    _graph_mode = []

    class GraphMode:
        STATIC = "static"
        DYNAMIC = "dynamic"
        DYNAMIC_CURR_NEXT = "dynamic_curr_next"

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
        dynamic_count = self.graph_mode.count(GraphIterator.GraphMode.DYNAMIC)
        dynamic_curr_next_count = self.graph_mode.count(GraphIterator.GraphMode.DYNAMIC_CURR_NEXT)
        if dynamic_count + dynamic_curr_next_count > 1:
            raise Exception("Incorrect GraphIterator mode configuration. Only one type of dynamic mode is allowed.")
        graphs = []
        for mode in self.graph_mode:
            if mode is self.GraphMode.DYNAMIC:
                graphs.append(GraphIterator.dynamic_graphs[self.current_id])
                self.current_id += 1
            elif mode is self.GraphMode.STATIC:
                graphs.append(GraphIterator.static_graph)
            elif mode is self.GraphMode.DYNAMIC_CURR_NEXT:
                graphs.extend([GraphIterator.dynamic_graphs[self.current_id],
                               GraphIterator.dynamic_graphs[self.current_id + 1]])
                self.current_id += 1
            if self.current_id >= len(GraphIterator.dynamic_graphs) \
                    or dynamic_curr_next_count > 0 and self.current_id + 1 >= len(GraphIterator.dynamic_graphs) \
                    or dynamic_count + dynamic_curr_next_count == 0:
                self.stop = True
        return graphs if len(graphs) > 1 else graphs[0]

        # if self.graph_mode is self.GraphMode.DYNAMIC or self.GraphMode is self.GraphMode.STATIC_DYNAMIC:
        #     graph = GraphIterator.dynamic_graphs[self.current_id]
        #     if self.current_id + 1 >= len(GraphIterator.dynamic_graphs):
        #         self.stop = True
        #     else:
        #         self.current_id += 1
        #     if self.GraphMode is self.GraphMode.STATIC_DYNAMIC:
        #         return graph, GraphIterator.static_graph
        #     return graph
        # if self.graph_mode is self.GraphMode.STATIC:
        #     self.stop = True
        #     return GraphIterator.static_graph
        # if self.graph_mode is self.GraphMode.ALL:
        #     self.graph_mode = self.GraphMode.DYNAMIC
        #     return GraphIterator.static_graph
        # if self.graph_mode is self.GraphMode.DYNAMIC_CURR_NEXT \
        #         or self.graph_mode is self.GraphMode.DYNAMIC_CURR_NEXT_AND_STATIC:
        #     graph = [GraphIterator.dynamic_graphs[self.current_id], GraphIterator.dynamic_graphs[self.current_id + 1]]
        #     if self.current_id + 2 >= len(GraphIterator.dynamic_graphs):
        #         self.stop = True
        #     else:
        #         self.current_id += 1
        #     if self.graph_mode is self.GraphMode.DYNAMIC_CURR_NEXT_AND_STATIC:
        #         graph.append(GraphIterator.static_graph)
        #     return graph

    def reset(self):
        self.stop = False
        self.current_id = 0
        self.graph_mode = self._graph_mode
