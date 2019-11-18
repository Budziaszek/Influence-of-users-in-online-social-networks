class CustomMetricsTypeDescription:
    def __init__(self, parent_metrics_1, operator, parent_metrics_2, graph_id_mod):
        self.parent_metrics_1 = parent_metrics_1
        self.parent_metrics_2 = parent_metrics_2
        self.operator = operator
        self.graph_id_mod = graph_id_mod

    def get_graph_id_for_parent_metric_2(self, basic_id):
        if self.graph_id_mod == "0":
            return 0
        if self.graph_id_mod == "-1":
            return basic_id - 1