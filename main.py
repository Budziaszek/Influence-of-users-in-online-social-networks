from Manager import Manager
from Network.GraphConnectionType import GraphConnectionType
from Mode import Mode

manager = Manager(parameters="dbname='salon24' user='sna_user' host='localhost' password='sna_password'")

modes_to_calculate = [
    Mode.comments_to_posts_from_others,
    Mode.comments_to_posts_and_comments_from_others,
    Mode.comments_to_comments_from_others
    # Mode.comments_to_posts,
    # Mode.comments_to_comments,
    #  Mode.comments_to_posts_and_comments,
]
values_to_calculate = [
    "neighbors_count",
    "connections_count",
    "connections_strength",
    "reciprocity"
]
connections_to_calculate = [
    GraphConnectionType.IN,
    GraphConnectionType.OUT
]

for mode in modes_to_calculate:
    manager.generate_graph_data(mode=mode, graph_type="sd", is_multi=False)
    for value in values_to_calculate:
        for connection in connections_to_calculate:
            manager.calculate_neighborhoods(calculated_value=value, connection_type=connection)
