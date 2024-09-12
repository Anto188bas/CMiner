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
t # 30
v 0 Access
v 1 BusinessProcess
v 2 BusinessObject
v 3 Access
v 4 BusinessProcess
v 5 Triggering
e 1 0 source
e 2 0 target
e 2 3 target
e 4 3 source
e 4 5 source
x (id-00ae4f79280548e4bf87f2d9999d727d, 2) (id-82442a4c03634d388a1bc2a24ed05a99, 692) (70ff1122-82b9-4d13-a15c-6cdc858dd4b3, 1) (id-c874bd6f-c66d-4b19-82bb-909e4f300ead, 30) (fa7c7a16-1338-4eae-97e2-7b9aed26799c, 5) (25eb1d65-b2a2-4123-ad8c-6578b468097d, 15) (id-cff0d7556f654881afbe90d896d2c969, 2) (id-ea12c7b9ede84a8dae334d9752053744, 21) (4d1d1da3-23ba-4562-a6c8-bca223c644fb, 3) (d35b7b41-b41e-417f-8f0b-4b46b25609b7, 442) (id-91edb5e7-9903-4a2c-867e-40db2b10cb5a, 271) (id-ee2bba95, 4) (54e87e47-f01c-49fe-af55-2068f4564bd2, 10) (c04bba8d-5b28-4f7d-957a-d0921f0d3142, 4) (265214de, 12) (id-723b6dc3-1c32-41a5-a02b-6d27b40c34d8, 64) (466b1e76, 9) (id-8d57243ff9f94eaeaad5033a98122432, 22) (d6c69f6a, 6) (id-316838d5-7351-46de-9071-f4ba8175825c, 16) (723b6dc3-1c32-41a5-a02b-6d27b40c34d8, 64) (b7b20375-d803-4baa-a59d-712acb1eae41, 92) (54525ee5, 255)

Support: 23
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
