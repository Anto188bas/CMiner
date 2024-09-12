from src.CMiner.CMiner import CMiner
import time

miner = CMiner("/Users/simoneavellino/Desktop/CMiner/Datasets/Archimate-dataset-100/graphs.data", min_num_nodes=5, max_num_nodes=6, support=0.2, approach='dfs', show_mappings=False)


start_time = time.time()
miner.mine()
end_time = time.time()
print(f"Mine - Execution time: {end_time - start_time} seconds")


# python -m gspan_mining -s 20 -d 1 -l 2 -u 10 /Users/simoneavellino/Desktop/CMiner/Datasets/OntoUML-dataset-143/graphs.data
