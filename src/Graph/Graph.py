import networkx as nx
import random


# TO DO: improve random query generation
# TO DO: come è strutturato l'arco esempio di tupla (source, target, key, id ) ?
# To DO: valutare assieme i metodi set_edge_attributes, are_equivalent_edge, compute_orbits_edge,edge_contain_attributes
# breaking_condition, e edge_id


def flat_map(list_of_lists):
    array = []
    for l in list_of_lists:
        array.extend(l)
    return array


NULL_LABEL = ""


# TO DO: valutare se ordinare le etichette o no
class MultiDiGraph(nx.MultiDiGraph):

    def __init__(self, incoming_graph_data=None, multigraph_input=None, **attr):
        super().__init__(incoming_graph_data, multigraph_input, **attr)
        self.node_labels = None
        self.edge_labels = None

    def reset_memoization(self):
        self.node_labels = None
        self.edge_labels = None

    def add_edge(self, u_of_edge, v_of_edge, key=None, **attr):
        if key is None:
            key = 0
            while self.has_edge(u_of_edge, v_of_edge, key):
                key += 1
        # check if the user pass the type attribute
        if 'type' not in attr:
            # set the type attribute to the default value
            attr['type'] = NULL_LABEL
        super().add_edge(u_of_edge, v_of_edge, key, **attr)

    def add_node(self, node_for_adding, **attr):
        if 'labels' not in attr:
            attr['labels'] = []
        super().add_node(node_for_adding, **attr)

    def edge_has_label(self, edge):
        source, target, key = edge
        return self[source][target][key]['type'] != NULL_LABEL

    def get_edge_labels(self, source, destination):
        labels = []
        if self.has_edge(source, destination):
            labels.extend([edge_data.get('type') for edge_data in self[source][destination].values() if edge_data.get('type') != NULL_LABEL])
        return sorted(set(labels))

    def get_edge_label(self, edge):
        source, destination, key = edge
        label = self[source][destination][key]['type']
        return self[source][destination][key]['type']

    def get_node_labels(self, id):
        return sorted(set(self.nodes[id]["labels"]))

    def get_all_node_labels(self):
        if self.node_labels is None:
            self.node_labels = sorted(set(flat_map([self.nodes[node]['labels'] for node in self.nodes])))
        return self.node_labels

    def get_all_edge_labels(self):
        if self.edge_labels is None:
            self.edge_labels = sorted(
                set([self.get_edge_data(edge[0], edge[1], edge[2])['type'] for edge in self.edges if self.get_edge_data(edge[0], edge[1], edge[2])['type'] != NULL_LABEL]))
        return self.edge_labels

    def get_edges_consider_no_direction(self, edge):
        """
        Returns the edges between source and destination nodes, not considering the direction of the edges.
        EXAMPLE:
            G = MultiDiGraph()
            G.add_edge(1, 2, type="A")
            G.add_edge(2, 1, type="B")
            G.get_edges_consider_no_direction(1, 2) -> [(1, 2, 0), (2, 1, 0)]
        :param edge: tuple (source, destination)
        :return list of edges between source and destination nodes
        """
        node_id_1, node_id_2 = edge
        edges = []
        if self.has_edge(node_id_1, node_id_2):
            edges.extend((node_id_1, node_id_2, key) for key in self[node_id_1][node_id_2])
        if self.has_edge(node_id_2, node_id_1):
            edges.extend((node_id_2, node_id_1, key) for key in self[node_id_2][node_id_1])
        return edges

    def edges_keys(self, edge):
        """
        Returns the edge keys between source and destination nodes.

        :param edge: tuple (source, destination)
        :return list of keys
        """
        if not self.has_edge(edge[0], edge[1]):
            return []
        return list(self[edge[0]][edge[1]].keys())

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

    def get_connected_subgraph_with_n_nodes(self, n):
        # Initialize the set of visited nodes
        visited = set()

        # Choose a random starting node
        start_node = random.choice(list(self.nodes()))

        # Initialize a queue for BFS
        queue = [start_node]

        # Explore the graph using BFS until finding a connected subgraph with n nodes
        while queue:
            node = queue.pop(0)
            visited.add(node)

            # If the connected subgraph has n nodes, return it
            if len(visited) == n:
                return self.subgraph(visited).copy()

            # Add neighboring nodes to the queue
            neighbors = list(self.neighbors(node))
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)

        # If no connected subgraph with n nodes is found, return None
        return None

    def get_all_edges(self):
        """
        Returns a list of tuples representing all edges in the graph,
        including their keys.
        Each tuple contains (source, target, key).
        """
        all_edges = []
        for u, v, key in self.edges(keys=True):
            if u == "dummy" or v == "dummy":
                continue
            all_edges.append((u, v, key))
        return all_edges

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

    def get_node_attributes(self, node_id):
        # delete labels from attributes
        return {k: v for k, v in self.nodes[node_id].items() if k != 'labels'}

    def get_edge_attributes(self, edge):
        # delete type from attributes
        source, target, key = edge
        return {k: v for k, v in self[source][target][key].items() if k != 'type'}

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

    def edge_contains_attributes(self, edge, attributes):
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
        source, target, key = edge
        return self.has_edge(source, target, key) and all(
            attr in self[source][target][key].items() for attr in attributes.items())

    # Valutare con Simone --->Funzionamento + Utility
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

    def compute_orbits_edges_id(self):
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

            orbits.append(list(orbit))

        return orbits

    def compute_orbits_nodes_id(self):
        orbits = []
        unvisited_nodes = set(self.nodes())

        while unvisited_nodes:
            start_node = unvisited_nodes.pop()
            orbit = {start_node}
            nodes_to_check = unvisited_nodes.copy()

            for node in nodes_to_check:
                if self.are_equivalent(start_node, node):
                    orbit.add(node)
                    unvisited_nodes.remove(node)

            orbits.append(list(orbit))

        return orbits

    @staticmethod
    def generate_graph(pattern, times, seed = None):
        """
        Generate a graph by repeating a given pattern multiple times.

        Args:
            pattern: The pattern graph to be repeated.
            times: The number of times the pattern should be repeated.

        Returns:
            A new graph containing 'times' repetitions of the pattern, connected to form a connected graph.
        """
        # Create a new MultiDiGraph
        mapping = {}
        new_graph = MultiDiGraph()

        if seed is not None:
            random.seed(seed)

        # Repeat the pattern 'times' times
        for i in range(times):
            # Add nodes and edges from the pattern to the new graph
            mapping = {}  # Mapping of old node IDs to new node IDs in the new graph
            for node in pattern.nodes(data=True):
                # Add node to the new graph
                new_node_id = new_graph.number_of_nodes()  # Generate unique node ID
                # Add node to the new graph considering all the attributes
                new_graph.add_node(new_node_id, **node[1])
                mapping[node[0]] = new_node_id

            for edge in pattern.edges(data=True, keys=True):
                # Add edge to the new graph
                source = mapping[edge[0]]
                target = mapping[edge[1]]
                new_graph.add_edge(source, target, key=edge[2], **edge[3])

            # Connect the pattern to the new graph
            if i > 0:
                # Create up to num_nodes random edges between pattern and new_graph
                num_new_edges = random.randint(1, len(pattern.nodes()))
                for _ in range(num_new_edges):
                    # take a random node from the pattern and a random node from the new graph
                    node_pattern = random.choice(list(pattern.nodes()))
                    # the new node on the graph cannot be the same as the node in the pattern (no loops)
                    cand = list(set(new_graph.nodes()).difference(set([mapping[n] for n in pattern.nodes()])))
                    node_new_graph = random.choice(cand)
                    # randomize the edge direction
                    if random.random() < 0.5:
                        src = node_new_graph
                        dest = mapping[node_pattern]
                    else:
                        src = mapping[node_pattern]
                        dest = node_new_graph
                    new_graph.add_edge(src, dest, type=random.choice(new_graph.get_all_edge_labels()))

        return new_graph