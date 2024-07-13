from src.CMiner.BitMatrix import QueryBitMatrixOptimized, TargetBitMatrixOptimized, BitMatrixStrategy2
from src.CMiner.BreakingConditions import BreakingConditionsNodes, BreakingConditionsEdges
from src.CMiner.CompatibilityDomain import CompatibilityDomainWithDictionary
from src.CMiner.Ordering import Ordering
<<<<<<< HEAD
import time
=======
import ray

>>>>>>> a5aebab156e565b1281d899a10ba7aa0f3777f21


class Solution:

    def __init__(self, f, g):
        self.f = f
        self.g = g

    def copy(self):
        return Solution(self.f.copy(), self.g.copy())

    def query_nodes(self):
        return self.f.keys()

    def query_edges(self):
        return self.g.keys()

    def target_nodes(self):
        return self.f.values()

    def target_edges(self):
        return self.g.values()

    def nodes_mapping(self):
        return self.f

    def add_node_mapping(self, node, target_node):
        self.f[node] = target_node

    def add_edge_mapping(self, edge, target_edge):
        self.g[edge] = target_edge

    def get_node_mapping(self, node):
        return self.f[node]

    def is_edge_mapped(self, edge):
        return edge in self.g.values()



    def __str__(self):
        str = "------------------------------------------\n"
        str += "Query node mapping:\n"
        for key in self.f.keys():
            str += f"{key} -> {self.f[key]}\n"
        str += "\nQuery edge mapping:\n"
        for key in self.g.keys():
            str += f"{key} -> {self.g[key]}\n"
        str += "------------------------------------------\n"
        return str


class MultiGraphMatch:

    def __init__(self, target, target_bit_matrix=None):
        self.target = target
        if target_bit_matrix is None:
            self.tbm = TargetBitMatrixOptimized(self.target, BitMatrixStrategy2())
        else:
            self.tbm = target_bit_matrix
        # all of this attributes are initialized in the match method
        # so that the class can be reused
        self.query = None
        self.qbm = None
        self.node_mapping_function = None
        self.edge_mapping_function = None
        self.cand = None
        self.cand_index = None
        self.br_cond_node = None
        self.br_cond_edge = None
        self.domain = None
        self.ordering = None
        self.solutions = None
        self.f = None
        self.g = None

    def _init_matching(self,
                       query,
                       query_bit_matrix=None,
                       compatibility_domain=None,
                       ordering=None,
                       breaking_conditions_nodes=None,
                       breaking_conditions_edges=None
                       ):
        self.solutions = []
        self.query = query
        self.f = {node: None for node in query.nodes()}
        self.g = {edge: None for edge in query.get_all_edges()}
        self.cand = {edge: [] for edge in self.query.get_all_edges()}
        self.cand_index = {edge: 0 for edge in self.query.get_all_edges()}
        if query_bit_matrix is None:
            self.qbm = QueryBitMatrixOptimized(query, BitMatrixStrategy2())
        else:
            self.qbm = query_bit_matrix
        if compatibility_domain is None:
            self.domain = CompatibilityDomainWithDictionary(self.qbm, self.tbm)
        else:
            self.domain = compatibility_domain
        if ordering is None:
            self.ordering = Ordering(self.query, self.domain)
        else:
            self.ordering = ordering
        if breaking_conditions_nodes is None:
            self.br_cond_node = BreakingConditionsNodes(self.query, self.f)
        else:
            self.br_cond_node = breaking_conditions_nodes
        if breaking_conditions_edges is None:
            self.br_cond_edge = BreakingConditionsEdges(self.query, self.g)
        else:
            self.br_cond_edge = breaking_conditions_edges


    def match(self,
              query,
              query_bit_matrix=None,
              compatibility_domain=None,
              ordering=None,
              breaking_conditions_nodes=None,
              breaking_conditions_edges=None):
        self._init_matching(query,
                            query_bit_matrix,
                            compatibility_domain,
                            ordering,
                            breaking_conditions_nodes,
                            breaking_conditions_edges)

        self.ordering.compute()
        forceBack = False
        i = 0
        q_i, q_j, q_key = query_edge = self.ordering.get(i)
        self._find_candidates(query_edge)
        while i >= 0:
            if forceBack or self.cand_index[query_edge] >= len(self.cand[query_edge]):
                forceBack = False
                if self.g[query_edge] is None:
                    i -= 1
                if i < 0:
                    # no more solutions
                    break
                q_i, q_j, q_key = query_edge = self.ordering.get(i)

                # backtracking
                # RESET THE MAPPING OF THE QUERY EDGE
                self.g[query_edge] = None
                # RESET THE MAPPING OF THE QUERY NODES
                # reset f[q_i] if there are no mapped edges in the query that have q_i as source/destination
                if all(self.g[q_e] is None for q_e in self.query.get_all_edges() if q_e[0] == q_i or q_e[1] == q_i):
                    self.f[q_i] = None
                # reset f[q_j] if there are no mapped edges in the query that have q_j as source/destination
                if all(self.g[q_e] is None for q_e in self.query.get_all_edges() if q_e[0] == q_j or q_e[1] == q_j):
                    self.f[q_j] = None
                self.cand_index[query_edge] += 1
            else:
                # CHECK IF THE NEXT CANDIDATE TARGET EDGE IS COMPATIBLE WITH THE QUERY EDGE
                # extract the target edge
                t_i, t_j, key = target_edge = self.cand[query_edge][self.cand_index[query_edge]]
                # check if the target edge has not been already mapped to another query edge
                if all(self.g[q_e] != target_edge for q_e in self.query.get_all_edges() if self.g[q_e] is not None):
                    # if the mapping for the source is not already set, set it
                    if self.f[q_i] is None:
                        self.f[q_i] = t_i
                    # if the mapping for the destination is not already set, set it
                    if self.f[q_j] is None:
                        self.f[q_j] = t_j
                    # set the mapping for the query edge
                    self.g[query_edge] = target_edge
                    # if the edge is the last in the ordering, the solution is found
                    if i == len(self.query.get_all_edges()) - 1:
                        forceBack = True
                        # SOLUTION FOUND
                        # save the mapping
                        self.solutions.append(Solution(self.f.copy(), self.g.copy()))
                    else:
                        # shift the index to the next query edge in the ordering
                        i += 1
                        # get the next query edge
                        q_i, q_j, q_key = query_edge = self.ordering.get(i)
                        # find the candidates for the next query edge
                        self._find_candidates(query_edge)
                        # reset the index for the candidates of the next query edge
                        self.cand_index[query_edge] = 0
                else:
                    # shift the index to the next candidate
                    self.cand_index[query_edge] += 1



    def _find_candidates(self, query_edge):
        q_i, q_j, query_key = query_edge
        self.cand[query_edge] = []
        if self.f[q_i] is None and self.f[q_j] is None:
            for t_i, t_j in self.domain.get_domain((q_i, q_j)):
                # each target edge is a tuple (source, target, edge_id)
                for target_key in self.target.edges_keys((t_i, t_j)):
                    target_edge = (t_i, t_j, target_key)
                    if (
                            (
                                    # if query edge label is not specified it means that any edge label is accepted
                                    not self.query.edge_has_label(query_edge) or
                                    # check if the edge labels are the same
                                    self.target.get_edge_label(target_edge) == self.query.get_edge_label(query_edge)
                            ) and
                            # check if the target edge source node contains the same attributes as the query edge source node
                            self.target.node_contains_attributes(t_i, self.query.get_node_attributes(q_i)) and
                            # check if the target edge destination node contains the same attributes as the query edge destination node
                            self.target.node_contains_attributes(t_j, self.query.get_node_attributes(q_j)) and
                            # check if the target edge contains the same attributes as the query edge
                            self.target.edge_contains_attributes(target_edge,
                                                                 self.query.get_edge_attributes(query_edge))
                    ):
                        self.cand[query_edge].append(target_edge)
        elif self.f[q_i] is not None and self.f[q_j] is not None:
            for target_key in self.target.edges_keys((self.f[q_i], self.f[q_j])):
                target_edge = (self.f[q_i], self.f[q_j], target_key)
                if (
                        (
                                # if query edge label is not specified it means that any edge label is accepted
                                not self.query.edge_has_label(query_edge) or
                                # check if the edge labels are the same
                                self.target.get_edge_label(target_edge) == self.query.get_edge_label(query_edge)
                        ) and
                        self.target.edge_contains_attributes(target_edge, self.query.get_edge_attributes(query_edge))
                ):
                    # if self.br_cond_edge.check(query_edge, target_edge):
                    self.cand[query_edge].append(target_edge)
        elif self.f[q_i] is not None:
            for t_i, t_j in self.domain.get_domain((q_i, q_j)):
                if t_i == self.f[q_i]:
                    for target_key in self.target.edges_keys((t_i, t_j)):
                        target_edge = (t_i, t_j, target_key)
                        if (
                                (
                                        # if query edge label is not specified it means that any edge label is accepted
                                        not self.query.edge_has_label(query_edge) or
                                        # check if the edge labels are the same
                                        self.target.get_edge_label(target_edge) == self.query.get_edge_label(query_edge)
                                ) and
                                self.target.node_contains_attributes(t_j, self.query.get_node_attributes(q_j)) and
                                self.target.edge_contains_attributes(target_edge,
                                                                     self.query.get_edge_attributes(query_edge))
                        ):
                            # if self.br_cond_node.check(q_j, t_j):
                            self.cand[query_edge].append(target_edge)
        else:
            for t_i, t_j in self.domain.get_domain((q_i, q_j)):
                if t_j == self.f[q_j]:
                    for target_key in self.target.edges_keys((t_i, t_j)):
                        target_edge = (t_i, t_j, target_key)

                        if (
                                self.target.get_edge_label(target_edge) == self.query.get_edge_label(query_edge) and
                                self.target.node_contains_attributes(t_i, self.query.get_node_attributes(q_i)) and
                                self.target.edge_contains_attributes(target_edge,
                                                                     self.query.get_edge_attributes(query_edge))
                        ):
                            # if self.br_cond_node.check(q_i, t_i):
                            self.cand[query_edge].append(target_edge)

    def get_solutions(self):
        return self.solutions


@ray.remote
def match_parallel_worker(target, query, worker_id):

    matcher = MultiGraphMatch(target, query)
    matcher.match(query)
    solutions = matcher.solutions()
    return solutions



