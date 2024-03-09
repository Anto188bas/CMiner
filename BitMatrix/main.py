# this file only shows how to compute the candidates using bitmatrix

import networkx as nx
from Graph import MultiDiGraph
from BitMatrix import TargetBitMatrix, QueryBitMatrix
from BitMatrixStrategy import BitMatrixStrategy1, BitMatrixStrategy2

# creating a toy target-graph
t = MultiDiGraph()
t.add_node(1, labels=['x', 'y', 'z'])
t.add_node(2, labels=['y'])
t.add_node(3, labels=['y', 'x'])
t.add_node(4, labels=['x'])

t.add_edge(2, 1, key = 'a', label='red')
t.add_edge(2, 1, key = 'b', label='white')
t.add_edge(1, 4, key = 'c', label='red')
t.add_edge(2, 4, key = 'd', label='white')
t.add_edge(4, 3, key = 'e', label='purple')
t.add_edge(4, 3, key = 'f', label='red')
t.add_edge(3, 4, key = 't', label='red')
t.add_edge(3, 2, key = 'h', label='red')
t.add_edge(3, 2, key = 'i', label='red')
t.add_edge(2, 3, key = 'j', label='purple')
t.add_edge(2, 3, key = 'k', label='white')

# creating a toy query-graph
q = MultiDiGraph()
q.add_node('a', labels=['y'])
q.add_node('b', labels=['y'])
q.add_edge('b', 'a', key = '1', label='red')
q.add_edge('b', 'a', key = '2', label='white')

bmt = TargetBitMatrix(t, BitMatrixStrategy1())
bmq = QueryBitMatrix(q, BitMatrixStrategy1())
candidates = bmq.find_candidates(bmt)
print(candidates)
