# Esempio di costruzione di un grafo con NetworkX
import networkx as nx
import pynauty

G = nx.Graph()
G.add_nodes_from([1, 2, 3])
G.add_edges_from([(1, 2), (2, 3)])

# Converti il grafo in un formato compatibile con PyNauty
nauty_graph = pynauty.Graph(number_of_vertices=3, directed=False, adjacency_dict={0: [1], 1: [0, 2], 2: [1]})
automorphisms = pynauty.autgrp(nauty_graph)
print(automorphisms)