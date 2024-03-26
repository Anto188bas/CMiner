import networkx as nx
import random
import matplotlib.pyplot as plt


""" Utility function
"""
def flat_map(list_of_lists):
    array = []
    for l in list_of_lists:
        array.extend(l)
    return array

#TO DO: valutare se ordinare le etichette o no
class MultiDiGraph(nx.MultiDiGraph):
    
    def __init__(self, incoming_graph_data=None, multigraph_input=None, **attr):
        super().__init__(incoming_graph_data, multigraph_input, **attr)
        self.node_labels = None
        self.edge_labels = None

    def reset_memoization(self):
        self.node_labels = None
        self.edge_labels = None
    
    def get_edge_labels(self, source, destination):
        labels = []
        if(self.has_edge(source, destination)):
            labels.extend([edge_data.get('type') for edge_data in self[source][destination].values()])
        return sorted(set(labels))
    
    def get_node_labels(self, id):
        return sorted(set(self.nodes[id]["labels"]))
    
    def get_all_node_labels(self):
        if self.node_labels == None:
            self.node_labels = sorted(set(flat_map([self.nodes[node]['labels'] for node in self.nodes])))
        return self.node_labels

    def get_all_edge_labels(self):
        if self.edge_labels is None:
            self.edge_labels = sorted(set([self.get_edge_data(edge[0], edge[1], edge[2])['type'] for edge in self.edges]))
        return self.edge_labels

    def generate_random_query(self, num_nodes, num_edges):
        """Generate a random query graph with num_nodes nodes and num_edges edges."""
        # Create a graph
        G = MultiDiGraph()

        # Add nodes with random labels
        for i in range(num_nodes):
            label = random.choice(self.get_all_node_labels())
            G.add_node(node_for_adding=i, labels=[label])

        # Add edges with random labels
        for _ in range(num_edges):
            u, v = random.sample(range(num_nodes), 2)
            label = random.choice(self.get_all_edge_labels())
            G.add_edge(u, v, type=label)

        return G