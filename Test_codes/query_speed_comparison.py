from   NetworkX.NetworkConfigurator import NetworkConfigurator
from   NetworkX.NetworksLoading     import NetworksLoading
from Graph.Graph import MultiDiGraph
from BitMatrix.BitMatrixStrategy import BitMatrixStrategy2
from BitMatrix.BitMatrix import QueryBitMatrix as QBM, TargetBitMatrix as TBM, QueryBitMatrixOptimized as QBMOpt, TargetBitMatrixOptimized as TBMOpt
import timeit
import matplotlib.pyplot as plt
import networkx as nx

type_file    = "data"
configurator = NetworkConfigurator(type_file)
networks     = NetworksLoading(type_file, configurator.config)
NUM_GRAPH_TO_CONSIDER = 100


def test(target_bm, query_bm):

    # computing bitmatrix
    start = timeit.default_timer()
    target_bm.compute()
    computing_time = timeit.default_timer() - start

    # matching
    start = timeit.default_timer()
    print(query_bm.find_candidates(target_bm))
    matching_time = timeit.default_timer() - start
    return {
        "computing_time": computing_time,
        "candidates_time": matching_time,
    }



stats = []
i = 0
for network_name, network_obj in networks.Networks.items():
    if i == NUM_GRAPH_TO_CONSIDER:
        break
    i += 1
    print("Testing: ", network_name)
    # create target graph
    target = MultiDiGraph(incoming_graph_data=network_obj)
    query = target.generate_random_query(50, 50)

    if len(target.get_all_edge_labels()) == 0:
        continue



    stats.append({
        "network_name": network_name,
        "standard_bitmatrix": test(TBM(target, BitMatrixStrategy2()), QBM(query, BitMatrixStrategy2())),
        "indexed_bitmatrix": test(TBMOpt(target, BitMatrixStrategy2()), QBMOpt(query, BitMatrixStrategy2()))
    })


def plot_stats(stats):
    standard_computing_times = []
    indexed_computing_times = []
    standard_candidates_times = []
    indexed_candidates_times = []

    for stat in stats:
        standard_computing_times.append(stat["standard_bitmatrix"]["computing_time"])
        indexed_computing_times.append(stat["indexed_bitmatrix"]["computing_time"])
        standard_candidates_times.append(stat["standard_bitmatrix"]["candidates_time"])
        indexed_candidates_times.append(stat["indexed_bitmatrix"]["candidates_time"])

    # Plotting computing time
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 3, 1)
    plt.plot(range(len(stats)), standard_computing_times, label='Standard BitMatrix')
    plt.plot(range(len(stats)), indexed_computing_times, label='Indexed BitMatrix')
    plt.xlabel('Network')
    plt.ylabel('Computing Time (seconds)')
    plt.title('Computing Time')
    plt.legend()

    # Plotting absolute difference in computing time
    plt.subplot(2, 3, 2)
    absolute_difference_computing = [indexed - standard for standard, indexed in zip(standard_computing_times, indexed_computing_times)]
    plt.plot(range(len(stats)), absolute_difference_computing, label='Absolute Difference')
    plt.xlabel('Network')
    plt.ylabel('indexed - standard of computing time (seconds)')
    plt.title('Time wasted in computing (less is better)')
    plt.legend()

    # Plotting candidate time
    plt.subplot(2, 3, 3)
    plt.plot(range(len(stats)), standard_candidates_times, label='Standard BitMatrix')
    plt.plot(range(len(stats)), indexed_candidates_times, label='Indexed BitMatrix')
    plt.xlabel('Network')
    plt.ylabel('Candidates Time (seconds)')
    plt.title('Candidates Time')
    plt.legend()

    # Plotting absolute difference in candidate time
    plt.subplot(2, 3, 4)
    absolute_difference_candidates = [standard - indexed for standard, indexed in zip(standard_candidates_times, indexed_candidates_times)]
    plt.plot(range(len(stats)), absolute_difference_candidates, label='Absolute Difference')
    plt.xlabel('Network')
    plt.ylabel('standard - indexed of candidates time (seconds)')
    plt.title('Time saved in finding candidates(more is better)')
    plt.legend()

    # Plotting time earned
    plt.subplot(2, 3, 5)
    time_earned = [saved - wasted for saved, wasted in zip(absolute_difference_candidates, absolute_difference_computing)]
    plt.plot(range(len(stats)), time_earned, label='Time Earned')
    plt.xlabel('Network')
    plt.ylabel('Time Earned (seconds)')
    plt.title('Time Earned (more is better)')
    plt.legend()

    # Set the same scale for all subplots
    axes = plt.gcf().axes
    for ax in axes:
        ax.set_ylim([min(min(standard_computing_times), min(indexed_computing_times)),
                     max(max(standard_computing_times), max(indexed_computing_times)) + 0.1])
        ax.set_xlim([0, len(stats)-1])

    plt.tight_layout()
    plt.show()

# Assuming 'stats' is your array of JSON data
plot_stats(stats)

tot_time_indexed = 0
tot_time_standard = 0
for stat in stats:
    tot_time_standard += stat["standard_bitmatrix"]["computing_time"] + stat["standard_bitmatrix"]["candidates_time"]
    tot_time_indexed += stat["indexed_bitmatrix"]["computing_time"] + stat["indexed_bitmatrix"]["candidates_time"]

print("Standard BitMatrix: ", tot_time_standard)
print("Indexed BitMatrix: ", tot_time_indexed)


