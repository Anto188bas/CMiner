import networkx as nx
import random

# TO-DO: test orbits with new graph 
# To-DO : sign all method with comment

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

    def get_edge_label(self, source, destination, key):
        return self[source][destination][key]['type']

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

    def get_edges_consider_no_direction(self, node_id_1, node_id_2):
        """
        Returns the edges between source and destination nodes, not considering the direction of the edges.
        EXAMPLE:
            G = MultiDiGraph()
            G.add_edge(1, 2, type="A")
            G.add_edge(2, 1, type="B")
            G.get_edges_consider_no_direction(1, 2) -> [(1, 2, 0), (2, 1, 0)]
        :param node_id_2:
        :param node_id_1:
        :return list of edges between source and destination nodes
        """
        edges = []
        if self.has_edge(node_id_1, node_id_2):
            edges.extend((node_id_1, node_id_2, key) for key in self[node_id_1][node_id_2])
        if self.has_edge(node_id_2, node_id_1):
            edges.extend((node_id_2, node_id_1, key) for key in self[node_id_2][node_id_1])
        return edges

    def edges_between_nodes(self, node_id_1, node_id_2):      # OPTIMIZE
        return [(u, v, key) for u, v, key in self.edges(keys=True) if
                (u, v) == (node_id_1, node_id_2) or (u, v) == (node_id_2, node_id_1)]

    def all_neighbors(self, node_id):
        """
        Returns all neighbors of the node with id node_id.
        It does not consider the direction of the edges.
        :param node_id: The ID of the node.
        :return: A list of neighbors of the node.
        """
        return set(self.successors(node_id)) | set(self.predecessors(node_id))

    def jaccard_similarity(self, node_id_1, node_id_2):
        """
        Compute the Jaccard similarity between two nodes considering all neighbors.
        The Jaccard similarity is defined as the size of the
        intersection of the neighbors of the two nodes divided
        by the size of the union of the neighbors of the two
        nodes.
        :param node_id_1:
        :param node_id_2:
        :return:
        """
        neighbors_1 = self.all_neighbors(node_id_1)
        neighbors_2 = self.all_neighbors(node_id_2)
        intersection = neighbors_1.intersection(neighbors_2)
        union = neighbors_1.union(neighbors_2)
        return len(intersection) / len(union)

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

    def tot_deg(self, node_id):
        """
        Returns the total degree of the node with id node_id.

        :param node_id: The ID of the node.
        :return: The total degree of the node.
        """
        return self.in_deg(node_id) + self.out_deg(node_id)

    def in_deg(self, node_id):
        """
        Returns the in-degree of the node with id node_id.

        :param node_id: The ID of the node.
        :return: The in-degree of the node.
        """
        return len(self.in_edges(node_id))

    def out_deg(self, node_id):
        """
        Returns the out-degree of the node with id node_id.

        :param node_id: The ID of the node.
        :return: The out-degree of the node.
        """
        return len(self.out_edges(node_id))

    def t_out_deg(self, node_id, t):
        """
        Returns the number of edges that exit from the node with id node_id and have label t.

        Parameters:
        - node_id: The ID of the node.
        - t: The label of the edge.

        Returns:
        - The number of edges that exit from the node with label t.
        """
        # Get all outgoing edges from the node with node_id
        out_edges = self.out_edges(node_id, data=True)

        # Count the number of edges with label t
        count = sum(1 for _, _, attrs in out_edges if attrs.get('type') == t)

        return count

    def t_in_deg(self, node_id, t):
        """
        Returns the number of edges that enter the node with id node_id and have label t.

        Parameters:
        - node_id: The ID of the node.
        - t: The label of the edge.

        Returns:
        - The number of edges that enter the node with label t.
        """
        # Get all incoming edges to the node with node_id
        in_edges = self.in_edges(node_id, data=True)

        # Count the number of edges with label t
        count = sum(1 for _, _, attrs in in_edges if attrs.get('type') == t)

        return count

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

    def compute_orbits_nodes(self):
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

    def set_node_attributes(self, node_attributes, attribute_name):
        for node, attributes in node_attributes.items():
            if node not in self.nodes:
                self.add_node(node)  # Aggiungi il nodo se non esiste
            self.nodes[node][attribute_name] = attributes

    # Metodo per impostare gli attributi degli archi
    def set_edge_attributes(self, edge_attributes, attribute_name):
        for edge, attribute in edge_attributes.items():
            u, v = edge  # Estrai i nodi sorgente e destinazione dall'arco
            if not self.has_edge(u, v):
                self.add_edge(u, v)  # Aggiungi l'arco se non esiste
            self[u][v][0][attribute_name] = attribute  # Imposta l'attributo per l'arco

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

    def node_contains_attributes(self, node_id, attributes):
        """
        Checks if a node in the graph contains the specified attributes.

        Args:
            node_id: The ID of the node to check.
            attributes: A dictionary containing the attributes to check for, 
                        where keys are attribute names and values are attribute values.

        Returns:
            True if the node contains all the specified attributes, False otherwise.
        """
        return node_id in self.nodes and all(attr in self.nodes[node_id].items() for attr in attributes.items())

    def edge_contains_attributes(self, source, target, attributes):
        """
        Checks if an edge in the graph contains the specified attributes.

        Args:
            source: The source node of the edge.
            target: The target node of the edge.
            attributes: A dictionary containing the attributes to check for, 
                        where keys are attribute names and values are attribute values.

        Returns:
            True if the edge contains all the specified attributes, False otherwise.
        """
        return self.has_edge(source, target) and all(attr in self[source][target][0].items() for attr in attributes.items())
    
    #Valutare con Simone --->Funzionamento + Utility 
    def compute_symmetry_breaking_conditions(self):
        """
        Computes the symmetry breaking conditions for both nodes and edges in the graph.

        Returns:
            A list containing the symmetry breaking conditions for nodes and edges.
            Each element in the list is a list of conditions for either nodes or edges,
            where each condition is represented as a list of node IDs or edge tuples.
        """
        # Compute orbits for nodes and edges
        node_orbits = self.compute_orbits_nodes()
        edge_orbits = self.compute_orbits_edges()

        # List to store the symmetry breaking conditions for nodes and edges
        breaking_conditions = []

        # Compute symmetry breaking conditions for nodes
        node_breaking_conditions = []
        for orbit in node_orbits:
            if len(orbit) > 1:
                smallest_node = min(orbit)
                # Sort the node IDs within each orbit for consistency
                condition = sorted(orbit, key=lambda node: node)
                node_breaking_conditions.append(condition)

        # Compute symmetry breaking conditions for edges
        edge_breaking_conditions = []
        for orbit in edge_orbits:
            if len(orbit) > 1:
                smallest_edge = min(orbit)
                # Sort the edge tuples within each orbit based on their third element (ID) for consistency
                condition = sorted(orbit, key=lambda edge: edge[2])
                edge_breaking_conditions.append(condition)

        # Append node and edge breaking conditions to the main list
        breaking_conditions.append(node_breaking_conditions)
        breaking_conditions.append(edge_breaking_conditions)

        return breaking_conditions

