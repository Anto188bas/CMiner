from parser import CMinerParser, gSpanParser
from comparator import *
from checker import Checker

db_path = ("/Users/simoneavellino/Desktop/Datasets/cycles-db/db1.data")
solutions_path = "/Users/simoneavellino/Desktop/sol_cycles1_s_2/cminer.txt"

checker = Checker(db_path, CMinerParser(solutions_path), "VF2")

checker.run()

#
# cminer_parser = CMinerParser("/Users/simoneavellino/Desktop/sol_cycles1_s_2/cminer.txt")
# gpsan_parser = gSpanParser("/Users/simoneavellino/Desktop/sol_cycles1_s_2/gspan.txt")
#
# Comparator(cminer_parser, gpsan_parser).solution_count().different_solutions()