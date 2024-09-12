import time
import argparse
from src.CMiner.CMiner_optimized import CMiner

def main_function():
    parser = argparse.ArgumentParser(description="CMiner algorithm")
    parser.add_argument('db_file', type=str, help="Path to graph db")
    parser.add_argument('support', type=float, help="Support")
    parser.add_argument('-l', '--min_nodes', type=int, help="Minimum number of nodes", default=1)
    parser.add_argument('-u', '--max_nodes', type=int, help="Maximum number of nodes", default=float('inf'))
    parser.add_argument('-m', '--show_mappings', type=int, help="Show pattern mappings", default=0)
    parser.add_argument('-o', '--output_path', type=str, help="Output file", default=None)
    parser.add_argument('-a', '--approach', type=str, help="Support", default='dfs')


    args = parser.parse_args()

    miner = CMiner(args.db_file, min_num_nodes=args.min_nodes, max_num_nodes=args.max_nodes, support=args.support,
                   show_mappings=args.show_mappings, output_path=args.output_path, approach=args.approach)

    start_time = time.time()
    miner.mine()
    end_time = time.time()
    print(f"\n-> Execution time: {end_time - start_time} seconds")



# pipreqs .
# pip install -e .