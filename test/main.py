from parser import CMinerParser, gSpanParser
from comparator import *
from checker import Checker

db_path = "/Users/simoneavellino/Desktop/CMiner/test/Datasets/OntoUML-db/graphs.data"
solutions_path = "/Users/simoneavellino/Desktop/CMiner/test/solution/a.txt"

checker = Checker(db_path, CMinerParser(solutions_path), matching_algorithm="VF2")

checker.isomorphic_solutions()




# cminer_parser = CMinerParser("/Users/simoneavellino/Desktop/CMiner/test/solution/cminer_sol.txt")
# gpsan_parser = gSpanParser("/Users/simoneavellino/Desktop/CMiner/test/solution/gspan_sol.txt")
#
# sol_60 = CMinerParser("/Users/simoneavellino/Desktop/CMiner/test/solution/no_match_48.data")
# sol_59 = CMinerParser("/Users/simoneavellino/Desktop/CMiner/test/solution/no_match_50.data")
#
#
# Comparator(sol_60, sol_59).different_solutions()