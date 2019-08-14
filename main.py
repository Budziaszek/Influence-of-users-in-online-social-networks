from Manager import Manager
from Network.GraphConnectionType import GraphConnectionType

manager = Manager(parameters="dbname='salon24' user='sna_user' host='localhost' password='sna_password'")
manager.generate_graph_data(mode="sd", is_multi=False)
print("Number of graphs created:", len(manager.graph_data))

# manager.calculate_neighborhoods(calculated_value="neighbors_count", connection_type=GraphConnectionType.IN_OUT)
# manager.calculate_neighborhoods(calculated_value="neighbors_count", connection_type=GraphConnectionType.IN)
# manager.calculate_neighborhoods(calculated_value="neighbors_count", connection_type=GraphConnectionType.OUT)
#
# manager.calculate_neighborhoods(calculated_value="connections_count", connection_type=GraphConnectionType.IN_OUT)
# manager.calculate_neighborhoods(calculated_value="connections_count", connection_type=GraphConnectionType.IN)
# manager.calculate_neighborhoods(calculated_value="connections_count", connection_type=GraphConnectionType.OUT)
