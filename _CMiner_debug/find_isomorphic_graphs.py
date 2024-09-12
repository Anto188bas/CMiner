import networkx as nx


def parse_graphs_from_file(filename):
    """
    Legge i grafi da un file e li restituisce in una lista.
    Ogni grafo è rappresentato come un oggetto networkx.Graph con etichette sui nodi e sugli archi.

    :param filename: Nome del file contenente i grafi
    :return: lista di grafi (oggetti networkx.Graph)
    """
    graphs = []
    current_graph = None
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()

            if line.startswith('t #'):
                # Nuovo grafo: inizializza un nuovo grafo
                if current_graph:
                    graphs.append(current_graph)
                current_graph = nx.Graph()

            elif line.startswith('v'):
                # Aggiungi un nodo al grafo
                _, node_id, label = line.split()
                current_graph.add_node(int(node_id), label=label)

            elif line.startswith('e'):
                # Aggiungi un arco al grafo
                _, node1, node2, label = line.split()
                current_graph.add_edge(int(node1), int(node2), label=label)

        # Aggiungi l'ultimo grafo dopo aver letto il file
        if current_graph:
            graphs.append(current_graph)

    return graphs


def check_isomorphism(graphs):
    """
    Controlla per ogni coppia di grafi in una lista se sono isomorfi,
    tenendo conto delle etichette sui nodi e sugli archi.

    :param graphs: lista di grafi networkx
    :return: lista di tuple con le coppie di grafi isomorfi
    """
    isomorphic_pairs = []

    # Funzione per confrontare le etichette dei nodi
    def node_match(n1, n2):
        return n1.get('label', None) == n2.get('label', None)

    # Funzione per confrontare le etichette degli archi
    def edge_match(e1, e2):
        return e1.get('label', None) == e2.get('label', None)

    # Confronta ogni coppia di grafi
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            G1 = graphs[i]
            G2 = graphs[j]

            # Usa la funzione is_isomorphic di networkx per controllare l'isomorfismo
            if nx.is_isomorphic(G1, G2, node_match=node_match, edge_match=edge_match):
                isomorphic_pairs.append((i, j))

    return isomorphic_pairs


# Esempio di utilizzo:
if __name__ == "__main__":
    # Leggi i grafi da un file (specifica il nome del file)
    filename = "mining_solutions.txt"  # Sostituisci con il nome corretto del file

    # Estrai i grafi dal file
    graphs = parse_graphs_from_file(filename)

    # Trova le coppie di grafi isomorfi
    result = check_isomorphism(graphs)

    # Stampa le coppie isomorfe
    if result:
        for (i, j) in result:
            print(f"I grafi con ID {i} e {j} sono isomorfi.")
    else:
        print("Nessun grafo isomorfo trovato.")
