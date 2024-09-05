import time
import argparse
from src.CMiner.CMiner_old import CMiner

parser = argparse.ArgumentParser(description="Pattern Mining Program")
parser.add_argument('db_file', type=str, help="Path to the input file")
parser.add_argument('-s', '--support', type=float, required=True, help="Support value")
parser.add_argument('-l', '--min_nodes', type=int, required=False, help="Minimum number of nodes", default=1)
parser.add_argument('-u', '--max_nodes', type=int, required=False, help="Maximum number of nodes", default=float('inf'))

args = parser.parse_args()

miner = CMiner(args.db_file, min_num_nodes=args.min_nodes, max_num_nodes=args.max_nodes, support=args.support)

start_time = time.time()
miner.mine()
end_time = time.time()
print(f"-> Execution time: {end_time - start_time} seconds")
