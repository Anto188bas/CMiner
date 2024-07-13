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

cc = 0
for network_name, network_obj in networks.Networks.items():
    g = MultiDiGraph(incoming_graph_data=network_obj)
    cc += len(g.compute_orbits_nodes())

print(cc / len(networks.Networks.items()))