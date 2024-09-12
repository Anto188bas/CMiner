from src.Graph import MultiDiGraph

g1 = MultiDiGraph()


# v 0 material
# v 1 1_1
# v 2 role
# v 3 role
# e 1 0 cardinalities
# e 2 0 target
# e 3 0 source
g1 = MultiDiGraph()
g1.add_node(0, labels=['material'])
g1.add_node(1, labels=['1_1'])
g1.add_node(2, labels=['role'])
g1.add_node(3, labels=['role'])
g1.add_edge(1, 0, type='cardinalities')
g1.add_edge(2, 0, type='target')
g1.add_edge(3, 0, type='source')


# v 0 material
# v 1 1_1
# v 2 role
# v 3 role
# e 1 0 cardinalities
# e 2 0 source
# e 3 0 target

g2 = MultiDiGraph()
g2.add_node(0, labels=['material'])
g2.add_node(1, labels=['1_1'])
g2.add_node(2, labels=['role'])
g2.add_node(3, labels=['role'])
g2.add_edge(1, 0, type='cardinalities')
g2.add_edge(2, 0, type='source')
g2.add_edge(3, 0, type='target')

print(g1.code())
print(g2.code())