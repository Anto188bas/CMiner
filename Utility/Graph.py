import networkx as nx
import random

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
            label = random.choice(self.get_all_node_labels())
            G.add_node(node_for_adding=i, labels=[label])

        # Add edges with random labels
        for _ in range(num_edges):
            u, v = random.sample(range(num_nodes), 2)
            label = random.choice(self.get_all_edge_labels())
            G.add_edge(u, v, type=label)

        return G

    def are_equivalent(node1, node2, G):
        # Verifica se le etichette di node1 sono un sottoinsieme o uguali alle etichette di node2
        if set(G.nodes[node1]['label']) == set(G.nodes[node2]['label']):
            # Verifica se gli attributi degli archi in uscita sono gli stessi

            out_edges_node1 = sorted([G.edges[edge]['type'] for edge in G.out_edges(node1)])
            out_edges_node2 = sorted([G.edges[edge]['type'] for edge in G.out_edges(node2)])

            if out_edges_node1 != out_edges_node2:
                return False

            # Verifica se gli attributi degli archi in entrata sono gli stessi
            in_edges_node1 = sorted([G.edges[edge]['type'] for edge in G.in_edges(node1)])
            in_edges_node2 = sorted([G.edges[edge]['type'] for edge in G.in_edges(node2)])

            if in_edges_node1 != in_edges_node2:
                return False
            return True
        else:

            return False

    def get_labels(self, graph):
        # Utilizza le funzionalità di networkx per ottenere le etichette dei nodi
        node_labels = nx.get_node_attributes(graph, 'label')

        # Dizionari per memorizzare le etichette degli archi uscenti e entranti
        edge_out_labels = {}
        edge_in_labels = {}

        # Itera attraverso tutti gli archi del grafo
        for source_node, target_node, edge_data in graph.edges(data=True):
            # Aggiunge l'etichetta dell'arco (tipo) al dizionario degli archi uscenti
            edge_out_labels.setdefault(source_node, {})[target_node] = edge_data.get('type', '')
            # Aggiunge l'etichetta dell'arco (tipo) al dizionario degli archi entranti
            edge_in_labels.setdefault(target_node, {})[source_node] = edge_data.get('type', '')

        # Restituisce i dizionari contenenti le etichette dei nodi, degli archi uscenti e degli archi entranti

        return node_labels, edge_out_labels, edge_in_labels

    def compute_orbits(self, graph):
        # Lista per memorizzare le orbite
        orbits = []

        # Insieme dei nodi non ancora visitati
        unvisited_nodes = set(graph.nodes())

        # Finché ci sono nodi non visitati
        while unvisited_nodes:
            # Prendi un nodo di partenza dalla lista dei nodi non visitati
            start_node = unvisited_nodes.pop()
            orbit = {start_node}

            # Copia di unvisited_nodes per iterare
            nodes_to_check = unvisited_nodes.copy()

            # Verifica l'equivalenza con gli altri nodi
            for node in nodes_to_check:
                if self.are_equivalent(start_node, node, graph):
                    orbit.add(node)
                    unvisited_nodes.remove(node)

            # Aggiungi l'orbita alla lista delle orbite
            orbits.append(orbit)
        # Restituisce la lista delle orbite
        return orbits

    def are_equivalent_edges(self, edge1, edge2, G):

        source1, target1 = edge1
        source2, target2 = edge2

        # Verifica se i nodi sorgente e destinazione hanno le stesse etichette
        if set(G.nodes[source1]['label']) == set(G.nodes[source2]['label']) and \
                set(G.nodes[target1]['label']) == set(G.nodes[target2]['label']):
            # Verifica se gli archi hanno lo stesso tipo
            if G.edges[edge1]['type'] == G.edges[edge2]['type']:
                return True

        return False

    def compute_orbits_edges(self, graph):

        orbits = []
        unvisited_edges = set(graph.edges())

        while unvisited_edges:
            start_edge = unvisited_edges.pop()
            orbit = {start_edge}
            edges_to_check = unvisited_edges.copy()

            for edge in edges_to_check:
                if self.are_equivalent_edges(start_edge, edge, graph):
                    orbit.add(edge)
                    unvisited_edges.remove(edge)

            orbits.append(orbit)

        return orbits

    def t_out_deg(self, node_id, t):
        return 1

    def t_in_deg(self, node_id, t):
        return 1