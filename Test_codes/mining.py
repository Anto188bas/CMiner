from Graph.Graph import MultiDiGraph

G = MultiDiGraph()

G.add_node(1, labels=["B"])
G.add_node(2, labels=["B"])
G.add_node(3, labels=["R"])
G.add_node(4, labels=["V"])
G.add_node(5, labels=["V"])
G.add_node(6, labels=["G"])
G.add_node(7, labels=["R"])

# G.add_node(8, labels=["B"])
# G.add_node(9, labels=["B"])
# G.add_node(10, labels=["B"])
# G.add_node(11, labels=["B"])
# G.add_node(12, labels=["B"])
#
# G.add_node(13, labels=["G"])
# G.add_node(14, labels=["G"])
# G.add_node(15, labels=["G"])
# G.add_node(16, labels=["G"])
# G.add_node(17, labels=["G"])


G.add_edge(7, 3, type="_")
G.add_edge(7, 5, type="_")
G.add_edge(7, 4, type="_")
G.add_edge(3, 1, type="_")
G.add_edge(5, 6, type="_")
G.add_edge(4, 2, type="_")

# G.add_edge(4, 8, type="_")
# G.add_edge(4, 9, type="_")
# G.add_edge(4, 10, type="_")
# G.add_edge(4, 11, type="_")
# G.add_edge(4, 12, type="_")
#
# G.add_edge(5, 13, type="_")
# G.add_edge(5, 14, type="_")
# G.add_edge(5, 15, type="_")
# G.add_edge(5, 16, type="_")
# G.add_edge(5, 17, type="_")


# G.add_edge(9, 9, type="_")


print(G.compute_orbits_nodes())



