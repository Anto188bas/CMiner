import numpy as np

from Graph.Graph import MultiDiGraph
from BitMatrix.BitMatrixStrategy import BitMatrixStrategy1, BitMatrixStrategy2
from   NetworkX.NetworkConfigurator import NetworkConfigurator
from   NetworkX.NetworksLoading     import NetworksLoading
from BitMatrix.BitMatrix import QueryBitMatrix as QBM, TargetBitMatrix as TBM, QueryBitMatrixOptimized as QBMOpt, TargetBitMatrixOptimized as TBMOpt




type_file    = "data"
configurator = NetworkConfigurator(type_file)
networks     = NetworksLoading(type_file, configurator.config)

# instantiate the BitMatrices for each graph
target_bms = []
infos = []
for network_name, network_obj in networks.Networks.items():
    print(network_name)
    target = MultiDiGraph(incoming_graph_data=network_obj)
    target_bm = TBMOpt(target, BitMatrixStrategy1())
    target_bm.compute()
    indices = target_bm.get_matrix_indices()
    matrix = [target_bm.split_bitmap_row(i) for i in range(len(indices))]
    l_first = [''.join('1' if bit else '0' for bit in row[0]) for row in matrix]
    l_second = [''.join('1' if bit else '0' for bit in row[3]) for row in matrix]
    t_in = [''.join('1' if bit else '0' for bit in row[1]) for row in matrix]
    t_out = [''.join('1' if bit else '0' for bit in row[2]) for row in matrix]
    infos.append({
        "network_name": network_name,
        "bitmap_entries": len(indices),
        "L_first_count_distinct_nodes": len(set(l_first)),
        "L_second_count_distinct_nodes": len(set(l_second)),
        "t_in_count_distinct_nodes": len(set(t_in)),
        "t_out_count_distinct_nodes": len(set(t_out))
    })


import matplotlib.pyplot as plt
# Extracting data for plotting
bitmap_entries = [info["bitmap_entries"] for info in infos]
L_first_count_distinct_nodes = [info["L_first_count_distinct_nodes"] for info in infos]
L_second_count_distinct_nodes = [info["L_second_count_distinct_nodes"] for info in infos]
T_count_distinct_nodes = [info["t_in_count_distinct_nodes"] for info in infos]
T_count_distinct_edges = [info["t_out_count_distinct_nodes"] for info in infos]

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plotting L_first_count_distinct_nodes
axs[0, 0].plot(bitmap_entries, L_first_count_distinct_nodes, 'o', label='L_first_count_distinct_nodes')
axs[0, 0].set_title('L_first bitmaps')
axs[0, 0].set_xlabel('BitMatrix entries')
axs[0, 0].set_ylabel('Count distinct labels')

# Plotting L_second_count_distinct_nodes
axs[0, 1].plot(bitmap_entries, L_second_count_distinct_nodes, 'o', label='L_second_count_distinct_nodes')
axs[0, 1].set_title('L_second bitmaps')
axs[0, 1].set_xlabel('BitMatrix entries')
axs[0, 1].set_ylabel('Count distinct labels')

# Plotting T_count_distinct_nodes
axs[1, 0].plot(bitmap_entries, T_count_distinct_nodes, 'o', label='t_in_count_distinct_nodes')
axs[1, 0].set_title('T_in bitmaps')
axs[1, 0].set_xlabel('BitMatrix entries')
axs[1, 0].set_ylabel('Count distinct labels')

# Plotting T_count_distinct_edges
axs[1, 1].plot(bitmap_entries, T_count_distinct_edges, 'o', label='t_out_count_distinct_nodes')
axs[1, 1].set_title('T_out bitmaps')
axs[1, 1].set_xlabel('BitMatrix entries')
axs[1, 1].set_ylabel('Count distinct labels')

fig.suptitle(f'Dataset containing {len(infos)} graphs')
# Adjust layout
plt.tight_layout()

plt.show()