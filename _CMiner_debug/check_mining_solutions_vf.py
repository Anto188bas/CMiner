from src.NetworkX.NetworkConfigurator import NetworkConfigurator
from src.NetworkX.NetworksLoading import NetworksLoading
from src.CMiner.MultiGraphMatch import MultiGraphMatch
from src.Graph.Graph import MultiDiGraph
from src.CMiner.BitMatrix import TargetBitMatrixOptimized, BitMatrixStrategy2
import networkx as nx


"""
!!!README:!!!
This file read the solutions find by CMiner and check if they are correct.

The solutions are read from the file mining_solutions.txt and the graphs are read from the database.
"""

def node_match(n1, n2):
    try:
        return sorted(n1['labels']) == sorted(n2['labels'])
    except KeyError:
        return False

def edge_match(e1, e2):
    return e1[list(e1.keys())[0]]['type'] == e2[0]['type']

class SolutionChecker:
    def __init__(self, solutions_file, type_file):
        self.solutions_file = solutions_file
        self.type_file = type_file
        self.configurator = NetworkConfigurator(self.type_file)
        self.networks = NetworksLoading(self.type_file, self.configurator.config)
        self.db_graphs = [MultiDiGraph(g) for g in self.networks.Networks.values()]
        self.db_graphs_names = list(self.networks.Networks.keys())
        self.all_node_labels = {label for g in self.db_graphs for label in g.get_all_node_labels()}
        self.all_edge_labels = {label for g in self.db_graphs for label in g.get_all_edge_labels()}

    def load_solutions(self):
        with open(self.solutions_file, "r") as f:
            solutions = f.read().split("-------------")
        return solutions

    def parse_solution(self, solution):
        graph = MultiDiGraph()
        nodes, edges, graphs_to_consider, frequencies = [], [], [], []
        solution_id = None

        for line in solution.strip().split("\n"):
            parts = line.split()
            if line.startswith("v"):
                _, node_id, label = parts
                nodes.append((node_id, label))
            elif line.startswith("e"):
                _, src, tgt, label = parts
                edges.append((src, tgt, label))
            elif line.startswith("x"):
                # remove last character (")")
                line = line[:-1]
                # Properly split the graphs and extract names
                graphs = line[3:].split(") (")
                for g in graphs:
                    if g:
                        graph_name, frequency = g.split(", ")
                        graphs_to_consider.append(graph_name)
                        frequencies.append(int(frequency))
            elif line.startswith("t"):
                _, _, graph_id = parts
                solution_id = graph_id
            elif line.startswith("f"):
                _, *frequencies = parts
                frequencies = [int(f) for f in frequencies]
            else:
                continue

        for node_id, label in nodes:
            graph.add_node(node_id, labels=[label])
        for src, tgt, label in edges:
            graph.add_edge(src, tgt, type=label)

        # Convert graph names to indices
        graphs_to_consider = [self.db_graphs_names.index(graph_name) for graph_name in graphs_to_consider]
        return graph, graphs_to_consider, solution_id, frequencies

    def check_solutions(self):
        wrong = 0
        solutions = self.load_solutions()

        for solution in solutions:
            solution_graph, graphs_to_consider, sol_id, frequencies = self.parse_solution(solution)
            print("- Checking solution", sol_id)
            # for each graph in the database search for the solution
            for idx_db_graph, db_graph in enumerate(self.db_graphs):

                matcher = nx.algorithms.isomorphism.GraphMatcher(db_graph, solution_graph, node_match=node_match,
                                                                 edge_match=edge_match)
                frequency = sum(1 for _ in matcher.subgraph_isomorphisms_iter())

                # if (idx_db_graph in graphs_to_consider and frequency != frequencies[graphs_to_consider.index(idx_db_graph)]) or (idx_db_graph not in graphs_to_consider and frequency > 0):

                if (idx_db_graph in graphs_to_consider and frequency == 0) or (idx_db_graph not in graphs_to_consider and frequency > 0):
                    print("    - Wrong solution for graph", self.db_graphs_names[idx_db_graph])
                    wrong += 1
                    break

        print("Accuracy:", (len(solutions) - wrong) / len(solutions) * 100, "%")





SOLUTIONS_FILE = "mining_solutions.txt"
TYPE_FILE = "data"
solution_checker = SolutionChecker(SOLUTIONS_FILE, TYPE_FILE)
solution_checker.check_solutions()
