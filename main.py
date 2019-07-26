from Manager import Manager

manager = Manager()
manager.generate_graph_data("s")
print("Number of graphs created:", len(manager.graph_data))
