from parser import CMinerParser, gSpanParser
from comparator import *
from checker import Checker

db_path = ("/Users/simoneavellino/Desktop/CMiner/test/Datasets/OntoUML-db/graphs.data")
solutions_path = "/Users/simoneavellino/Desktop/sol_onto_20.txt"

checker = Checker(db_path, CMinerParser(solutions_path))

checker.run()

#
# cminer_parser = CMinerParser("/Users/simoneavellino/Desktop/sol_cycles1_s_2/cminer.txt")
# gpsan_parser = gSpanParser("/Users/simoneavellino/Desktop/sol_cycles1_s_2/gspan.txt")
#
# Comparator(cminer_parser, gpsan_parser).solution_count().different_solutions()