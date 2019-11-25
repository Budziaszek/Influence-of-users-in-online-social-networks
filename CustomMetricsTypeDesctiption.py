class CustomMetricsTypeDescription:
    def __init__(self, name, parent_metrics_1, operator, parent_metrics_2, graph_id_mod_1=0, graph_id_mod_2=0):
        self.name = name
        self.parent_metrics_1 = parent_metrics_1
        self.parent_metrics_2 = parent_metrics_2
        self.operator = operator
        self.graph_id_mod_1 = graph_id_mod_1
        self.graph_id_mod_2 = graph_id_mod_2
