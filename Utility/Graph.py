import networkx as nx
import random

# TO-DO: test orbits with new graph 
#To-DO : sign all method with comment 

""" Utility function
"""


def flat_map(list_of_lists):
    array = []
    for l in list_of_lists:
        array.extend(l)
    return array


# TO DO: valutare se ordinare le etichette o no
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
        if (self.has_edge(source, destination)):
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
            self.edge_labels = sorted(
                set([self.get_edge_data(edge[0], edge[1], edge[2])['type'] for edge in self.edges]))
        return self.edge_labels

    def generate_random_query(self, num_nodes, num_edges):
        # Create a graph
        G = MultiDiGraph()

        # Add nodes with random labels
        for i in range(num_nodes):
            labels = random.choice(self.get_all_node_labels())
            self.add_node(node_for_adding=i, labels=[labels])

        # Add edges with random labels
        for _ in range(num_edges):
            u, v = random.sample(range(num_nodes), 2)
            labels = random.choice(self.get_all_edge_labels())
            self.add_edge(u, v, type=labels)

        return G

    def are_equivalent(self, node1, node2):
        
        # Verifica se le etichette di node1 sono un sottoinsieme o uguali alle etichette di node2
        if set(self.nodes[node1]['labels']) == set(self.nodes[node2]['labels']):
            # Verifica se gli attributi degli archi in uscita sono gli stessi
            out_edges_node1 = sorted([self.edges[edge]['type'] for edge in self.out_edges(node1, keys=True)])
            out_edges_node2 = sorted([self.edges[edge]['type'] for edge in self.out_edges(node2, keys=True)])
            if out_edges_node1 != out_edges_node2:  
                return False

            # Verifica se gli attributi degli archi in entrata sono gli stessi
            in_edges_node1 = sorted([self.edges[edge]['type'] for edge in self.in_edges(node1, keys=True)])
            in_edges_node2 = sorted([self.edges[edge]['type'] for edge in self.in_edges(node2, keys=True)])
            if in_edges_node1 != in_edges_node2:
                return False
            # Se tutte le condizioni sono soddisfatte, i nodi sono equivalenti
            return True
        else:
            return False


    def compute_orbits(self):
        # Lista per memorizzare le orbite
        orbits = []
        
        # Insieme dei nodi non ancora visitati
        unvisited_nodes = set(self.nodes())
        
        # Finché ci sono nodi non visitati
        while unvisited_nodes:
            # Prendi un nodo di partenza dalla lista dei nodi non visitati
            start_node = unvisited_nodes.pop()
            orbit = {start_node}
            
            # Copia di unvisited_nodes per iterare
            nodes_to_check = unvisited_nodes.copy()
            
            # Verifica l'equivalenza con gli altri nodi
            for node in nodes_to_check:
                if self.are_equivalent(start_node, node):
                    orbit.add(node)
                    unvisited_nodes.remove(node)
            # Aggiungi l'orbita alla lista delle orbite
            orbits.append(orbit)

        return orbits

    def are_equivalent_edges(self, edge1, edge2):

        source1, target1, key1 = edge1
        source2, target2, key2 = edge2

        # Verifica se i nodi sorgente e destinazione hanno le stesse etichette
        if (set(self.nodes[source1]['labels']) == set(self.nodes[source2]['labels'])) and \
                (set(self.nodes[target1]['labels']) == set(self.nodes[target2]['labels'])):
            # Verifica se gli archi hanno lo stesso tipo
            if self.edges[edge1]['type'] == self.edges[edge2]['type']:
                return True

        return False


    def compute_orbits_edges(self):

        orbits = []
        unvisited_edges = set(self.edges(keys=True))

        while unvisited_edges:
            start_edge = unvisited_edges.pop()
            
            orbit = {start_edge}
            edges_to_check = unvisited_edges.copy()

            for edge in edges_to_check:
                if self.are_equivalent_edges(start_edge, edge):
                    orbit.add(edge)
                    unvisited_edges.remove(edge)
                    
            orbits.append(orbit)
            
        return orbits

    def t_out_deg(self, node_id, t):
        return 1

    def t_in_deg(self, node_id, t):
        return 1
