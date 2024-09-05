CMiner configuration
CMiner("", min_num_nodes=5, max_num_nodes=5, support=70)

Consider only graph "aristotle-ontology2019.json"
In this graph the matching finds 80 solutions, while the mining 78.
GRAPH:
Q = MultiDiGraph()
Q.add_node(1, labels=["kind"])
Q.add_node(2, labels=["gen"])
Q.add_node(3, labels=["gen"])
Q.add_node(4, labels=["subkind"])
Q.add_node(5, labels=["subkind"])

Q.add_edge(1, 2, type="general")
Q.add_edge(1, 3, type="general")
Q.add_edge(4, 2, type="specific")
Q.add_edge(5, 3, type="specific")




