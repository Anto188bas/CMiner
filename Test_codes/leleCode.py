import networkx as nx
import random
import matplotlib.pyplot as plt


class MultiDiGraph(nx.MultiDiGraph):

    def __init__(self, incoming_graph_data=None, multigraph_input=None, **attr):
        super().__init__(incoming_graph_data, multigraph_input, **attr)
        self.node_labels = None
        self.edge_labels = None

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

        # FinchÃ© ci sono nodi non visitati
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

    def compute_node_breaking_conditions(self):
        # Calcola le condizioni di rottura della simmetria per i nodi del grafo
        breaking_conditions = {}

        # Calcoliamo le orbite degli archi
        orbits_edges = self.compute_orbits_edges_id()

        # Creiamo un dizionario per associare ogni nodo agli ID degli archi che lo coinvolgono
        node_to_edge_ids = {node: set() for node in self.nodes()}
        for orbit in orbits_edges:
            for edge in orbit:
                for node in edge[:2]:
                    node_to_edge_ids[node].add(edge)  # Utilizziamo edge invece di orbit

        # Per ogni nodo, creiamo un insieme di tutti gli ID dei nodi che lo coinvolgono
        for node, edge_ids in node_to_edge_ids.items():
            # Estraiamo gli ID dei nodi dagli ID degli archi
            node_ids = {node_id for edge in edge_ids for node_id in edge[:2]}
            # Ordiniamo gli ID dei nodi
            sorted_node_ids = sorted(node_ids)
            # Creiamo una stringa rappresentante gli ID dei nodi ordinati
            breaking_condition = ''.join(str(node_id) for node_id in sorted_node_ids)
            # Assegniamo la condizione di rottura al nodo
            breaking_conditions[node] = breaking_condition

        return breaking_conditions

    def compute_symmetry_breaking_conditions(self):
        """
        Computes the symmetry breaking conditions for both nodes and edges in the graph.

        Returns:
            A list containing the symmetry breaking conditions for nodes and edges.
            Each element in the list is a list of conditions for either nodes or edges,
            where each condition is represented as a tuple of node IDs or edge tuples.
        """
        # Compute orbits for nodes and edges
        node_orbits = self.compute_orbits_nodes_id()
        edge_orbits = self.compute_orbits_edges_id()

        # List to store the symmetry breaking conditions for nodes and edges
        breaking_conditions = []

        # Compute symmetry breaking conditions for nodes
        node_breaking_conditions = []
        for orbit in node_orbits:
            if len(orbit) > 1:
                # Sort node IDs within each orbit for consistency
                sorted_orbit = sorted(orbit)
                smallest_node = min(sorted_orbit)
                condition = (sorted_orbit, smallest_node)
                node_breaking_conditions.append(condition)

        # Compute symmetry breaking conditions for edges
        edge_breaking_conditions = []
        for orbit in edge_orbits:
            if len(orbit) > 1:
                # Sort edge tuples within each orbit based on their IDs
                sorted_orbit = sorted(orbit)
                smallest_edge = min(sorted_orbit)  # Considera il minimo dell'orbita
                condition = (sorted_orbit, smallest_edge)
                edge_breaking_conditions.append(condition)

        # Append node and edge breaking conditions to the main list
        breaking_conditions.append(node_breaking_conditions)
        breaking_conditions.append(edge_breaking_conditions)

        return breaking_conditions

    def calculate_automorphisms_and_orbits(self):
        """
        Calculate automorphisms and orbits of the graph.

        Returns:
            A tuple containing a dictionary of automorphisms and a list of orbits.
        """
        automorphisms = {}
        orbits = []
        visited = set()

        for node in self.nodes():
            if node not in visited:
                orbit = {node}
                visited.add(node)
                for neighbor in self.neighbors(node):
                    if neighbor not in visited:
                        orbit.add(neighbor)
                        visited.add(neighbor)
                orbits.append(orbit)

                # Assign the same automorphism value to all nodes in the orbit
                for n in orbit:
                    automorphisms[n] = len(orbits) - 1

        return automorphisms, orbits

    def apply_breaking_conditions(orbit, node_labels, edge_out_labels, edge_in_labels):
        """
        Apply symmetry breaking conditions to the given orbit.

        Args:
            orbit: A set of nodes in the orbit.
            node_labels: A dictionary containing labels for nodes.
            edge_out_labels: A dictionary containing labels for outgoing edges.
            edge_in_labels: A dictionary containing labels for incoming edges.

        Returns:
            The filtered orbit after applying symmetry breaking conditions.
        """
        min_label_nodes = {}

        for node in orbit:
            min_label_node = node
            for other_node in orbit:
                if node != other_node:
                    if node_labels[other_node] < node_labels[min_label_node]:
                        min_label_node = other_node
            min_label_nodes[min_label_node] = True

        filtered_orbit = {node for node in orbit if node in min_label_nodes}

        return filtered_orbit

    def apply_breaking_conditions_by_id(orbit, node_attributes):
        """
        Apply symmetry breaking conditions to the given orbit based on node IDs.

        Args:
            orbit: A set of nodes in the orbit.
            node_attributes: A dictionary containing attributes for nodes.

        Returns:
            The filtered orbit after applying symmetry breaking conditions.
        """
        min_id_nodes = {}

        for node in orbit:
            min_id_node = node
            for other_node in orbit:
                if node != other_node:
                    if node_attributes[other_node]['id'] < node_attributes[min_id_node]['id']:
                        min_id_node = other_node
            min_id_nodes[min_id_node] = True

        filtered_orbit = {node for node in orbit if node in min_id_nodes}

        return filtered_orbit


def main():
    # Creazione di un grafo complesso di esempio
    G = MultiDiGraph()
    G.add_nodes_from([(1, {'labels': ['B']}), (4, {'labels': ['A']}), (3, {'labels': ['A']})])
    G.add_edges_from([(1, 4, 0, {'type': 'X'}), (1, 3, 0, {'type': 'X'})])

    # Calcolo e stampa delle orbite dei nodi
    orbits = G.compute_orbits_nodes_id()
    print("Orbite dei nodi:")
    for i, orbit in enumerate(orbits):
        print(f"Orbita {i + 1}: {orbit}")

    # Calcolo e stampa delle orbite degli archi
    orbits_edges = G.compute_orbits_edges_id()
    print("\nOrbite degli archi:")
    for i, orbit in enumerate(orbits_edges):
        print(f"Orbita {i + 1}: {orbit}")

    # Calcolo e stampa degli automorfismi e delle orbite
    automorphisms, orbits = G.calculate_automorphisms_and_orbits()
    print("\nAutomorfismi:")
    print(automorphisms)

    # Calcolo e stampa delle condizioni di rottura della simmetria
    breaking_conditions = G.compute_symmetry_breaking_conditions()
    print("\nCondizioni di rottura della simmetria:")
    for i, condition in enumerate(breaking_conditions):
        print(f"Condizioni per {'nodi' if i == 0 else 'archi'}:")
        for j, item in enumerate(condition):
            print(f"Orbita {j + 1}: {item}")


if __name__ == "__main__":
    main()

