from src.CMiner.CMiner import CMiner
import time

miner = CMiner("", min_num_nodes=1, max_num_nodes=10, support=20, output_file_name="output.txt")


start_time = time.time()
miner.mine()
end_time = time.time()
print(f"Mine - Execution time: {end_time - start_time} seconds")


# python -m gspan_mining -s 20 -d 1 -l 2 -u 10 /Users/simoneavellino/Desktop/CMiner/Datasets/OntoUML-dataset-143/graphs.data


