from Manager import Manager

manager = Manager(parameters="dbname='salon24' user='sna_user' host='localhost' password='sna_password'")
manager.generate_graph_data(mode="s", is_multi=False)
print("Number of graphs created:", len(manager.graph_data))

print("Edges weights:")
for node1, node2, data in manager.graph_data[0].G.edges(data=True):
    print(data['weight'])
