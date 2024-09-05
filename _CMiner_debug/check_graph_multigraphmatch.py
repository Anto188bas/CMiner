from src.NetworkX.NetworkConfigurator import NetworkConfigurator
from src.NetworkX.NetworksLoading import NetworksLoading
from src.CMiner.MultiGraphMatch import MultiGraphMatch
from src.Graph.Graph import MultiDiGraph
import timeit

type_file = "data"
configurator = NetworkConfigurator(type_file)
networks = NetworksLoading(type_file, configurator.config)
NUM_GRAPH_TO_CONSIDER = 1

# retrieve graphs from the database
db = networks.Networks.items()
db_names = list(networks.Networks.keys())
db_graphs = [MultiDiGraph(g) for g in networks.Networks.values()]
db_len = len(db_graphs)


def parse_solution(solution):
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

    for node_id, label in nodes:
        graph.add_node(node_id, labels=[label])
    for src, tgt, label in edges:
        graph.add_edge(src, tgt, type=label)

    # Convert graph names to indices
    graphs_to_consider = [db_names.index(graph_name) for graph_name in graphs_to_consider]
    return graph, graphs_to_consider, solution_id, frequencies


solution_str = """
t # 212
v 1 gen
v 2 kind
v 3 gen
v 4 category
v 5 gen
v 6 gen
e 2 1 general
e 2 3 specific
e 4 3 general
e 4 5 general
e 4 6 general
x (digitaldoctor2022.json, 392) (van-ee2021modular.json, 24) (ppo-o2021.json, 2) (silveira2021oap.json, 8) (qam.json, 6) (alpinebits2022.json, 300) (srro-ontology.json, 84) (aguiar2018rdbs-o.json, 60) (aires2022valuenetworks-geo.json, 12) (goncalves2011ecg.json, 320) (fonseca2022incorporating.json, 36) (guizzardi2005ontological.json, 20) (bernasconi2021ontovcm.json, 24) (bernasconi2023fair-principles.json, 20) (aristotle-ontology2019.json, 18) (sikora2021online-education.json, 2) (mgic-antt2011.json, 18350) (barcelos2015transport-networks.json, 6434) (albuquerque2011ontobio.json, 48) (cmpo2017.json, 336) (silva2012itarchitecture.json, 454) (lindeberg2022full-ontorights.json, 72) (dpo2017.json, 210) (kritz2020ontobg.json, 76) (derave2019dpo.json, 78) (spo2017.json, 60) (abrahao2018agriculture-operations.json, 12) (tourbo2021.json, 20) (lindeberg2022simple-ontorights.json, 30) (aguiar2019ooco.json, 30)

Support: 30
"""

Q = parse_solution(solution_str)[0]

# uniforming all bit matrices
all_node_labels = set()
all_edge_labels = set()
for i in range(db_len):
    all_node_labels = all_node_labels.union(db_graphs[i].get_all_node_labels())
    all_edge_labels = all_edge_labels.union(db_graphs[i].get_all_edge_labels())
for i in range(db_len):
    missing_node_labels = all_node_labels.difference(db_graphs[i].get_all_node_labels())
    missing_edge_labels = all_edge_labels.difference(db_graphs[i].get_all_edge_labels())
    db_graphs[i].add_node('dummy', labels=missing_node_labels)
    for label in missing_edge_labels:
        db_graphs[i].add_edge('dummy', 'dummy', type=label)

support = 10
count = 0
print("Number of graphs in the database: ", db_len)
found_graphs = []
for i, target in enumerate(db_graphs):
    start_time = timeit.default_timer()
    msm = MultiGraphMatch(target)
    msm.match(Q)
    sol_count = len(msm.get_solutions())
    if sol_count > 0:
        print("Graph ", db_names[i], " - Count: ", sol_count)
        found_graphs.append(i)
        count += 1
    end_time = timeit.default_timer()

print("\n\nPresent in ", count, " graphs out of ", db_len, " graphs")

# print(" ".join([db_names[i] for i in found_graphs]))
