from parser import CMinerParser, gSpanParser
from comparator import *
from checker import Checker

db_path = "/Users/simoneavellino/Desktop/Datasets/gspan/graphs.data"
solutions_path = "/Users/simoneavellino/Desktop/solutions/db-gspan_different_sol_cminer_400.txt"

checker = Checker(db_path, CMinerParser(solutions_path), "VF2")

checker.run()


# cminer_parser = CMinerParser("/Users/simoneavellino/Desktop/solutions/db-gspan_cminer_400.txt")
# gpsan_parser = gSpanParser("/Users/simoneavellino/Desktop/solutions/db-gspan_gspan_400.txt")
#
# Comparator(cminer_parser, gpsan_parser).different_solutions(algorithm_y=False)
