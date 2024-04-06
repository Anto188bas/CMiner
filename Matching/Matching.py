from BitMatrix.BitMatrix import QueryBitMatrixOptimized, TargetBitMatrixOptimized
from BitMatrix.BitMatrixStrategy import BitMatrixStrategy2
from BreakingConditions.BreakingConditions import BreakingConditionsNodes, BreakingConditionsEdges
from CompatibilityDomain.CompatibilityDomain import CompatibilityDomainWithDictionary
from Ordering.Ordering import Ordering

class MultiSubgraphMatching:

    def __init__(self, target, query):
        self.target = target
        self.query = query
        qbm = QueryBitMatrixOptimized(self.query, BitMatrixStrategy2())
        tbm = TargetBitMatrixOptimized(self.target, BitMatrixStrategy2())
        self.f = {node: None for node in self.query.nodes()}
        self.g = {edge: None for edge in self.query.get_all_edges()}
        self.cand = {edge: [] for edge in self.query.get_all_edges()}
        self.cand_index = {edge: 0 for edge in self.query.get_all_edges()}
        self.br_cond_node = BreakingConditionsNodes(self.query, self.f)
        self.br_cond_edge = BreakingConditionsEdges(self.query, self.g)
        self.domain = CompatibilityDomainWithDictionary(qbm, tbm)
        self.ordering = Ordering(self.query, self.domain)
        self.occurrences = []

    def get_occurrences(self):
        return self.occurrences

    def match(self):
        self.ordering.compute()
        i = 0
        q_i, q_j, q_key = query_edge = self.ordering.get(i)
        self._find_candidates(query_edge)
        print("Cand(e_q)", query_edge)
        while i >= 0:

            if self.cand_index[query_edge] >= len(self.cand[query_edge]):
                print("CANDIDATES EXHAUSTED, BACKTRACKING")
                # backtraking
                print("  -Reset g[", query_edge, "] = None")
                self.g[query_edge] = None
                # reset f values
                # reset f[q_i] if there are no edges in the query that have q_i as source/destination
                # and are mapped to a target edge
                if all(self.g[q_e] is None for q_e in self.query.get_all_edges() if q_e[0] == q_i or q_e[1] == q_i):
                    print("  -Reset f[", q_i, "]")
                    self.f[q_i] = None
                # reset f[q_j] if there are no edges in the query that have q_j as source/destination
                # and are mapped to a target edge
                if all(self.g[q_e] is None for q_e in self.query.get_all_edges() if q_e[0] == q_j or q_e[1] == q_j):
                    print("  -Reset f[", q_j, "]")
                    self.f[q_j] = None



                i -= 1
                if i < 0:
                    break
                q_i, q_j, q_key = query_edge = self.ordering.get(i)
                print("  -Cand(e_q)", query_edge)
            else:
                print("NEXT CANDIDATE")
                t_i, t_j, key = target_edge = self.cand[query_edge][self.cand_index[query_edge]]
                print("  -Cand(e_t):", target_edge)
                if all(self.g[q_e] != target_edge for q_e in self.query.get_all_edges() if self.g[q_e] is not None):
                    print(" Target edge mappable")
                    if self.f[q_i] is None:
                        print("  -f[", q_i, "] = ", t_i)
                        self.f[q_i] = t_i
                    if self.f[q_j] is None:
                        print("  -f[", q_j, "] = ", t_j)
                        self.f[q_j] = t_j
                    print("  -g[", query_edge, "] = ", target_edge)
                    self.g[query_edge] = target_edge
                    if i == len(self.query.get_all_edges()) - 1:
                        print("!!!!!!!!!!!!!!!!!!!!!!SOLUTION FOUND!!!!!!!!!!!!!!!!!!!!!!")
                        # solution found
                        self.occurrences.append(self.g.copy())
                        self.cand_index[query_edge] += 1
                    else:
                        print("NEXT QUERY EDGE")
                        i += 1
                        q_i, q_j, q_key = query_edge = self.ordering.get(i)
                        print("Cand(e_q)", query_edge)
                        self._find_candidates(query_edge)
                        self.cand_index[query_edge] = 0
                else:
                    print("Target edge not mappable")
                    self.cand_index[query_edge] += 1




    def _find_candidates(self, query_edge):
        print("Find candidates")
        q_i, q_j, query_id = query_edge
        self.cand[query_edge] = []
        if self.f[q_i] is None and self.f[q_j] is None:
            print("Source and destinaions not mapped")
            for t_i, t_j in self.domain.get_domain((q_i, q_j)):
                # each target edge is a tuple (source, target, edge_id)
                for target_key in self.target.edges_keys((t_i, t_j)):
                    target_edge = (t_i, t_j, target_key)
                    if (
                            # check if the edge labels are the same
                            self.target.get_edge_label(target_edge) == self.query.get_edge_label(query_edge) and
                            # check if the target edge source node contains the same attributes as the query edge source node
                            self.target.node_contains_attributes(t_i, self.query.get_node_attributes(q_i)) and
                            # check if the target edge destination node contains the same attributes as the query edge destination node
                            self.target.node_contains_attributes(t_j, self.query.get_node_attributes(q_j)) and
                            # check if the target edge contains the same attributes as the query edge
                            self.target.edge_contains_attributes(target_edge, self.query.get_edge_attributes(query_edge))
                    ):
                        self.cand[query_edge].append(target_edge)
        elif self.f[q_i] is not None and self.f[q_j] is not None:
            print("Source and destinaions mapped")
            for target_key in self.target.edges_keys((self.f[q_i], self.f[q_j])):
                target_edge = (self.f[q_i], self.f[q_j], target_key)
                if (
                        self.target.get_edge_label(target_edge) == self.query.get_edge_label(query_edge) and
                        self.target.edge_contains_attributes(target_edge, self.query.get_edge_attributes(query_edge))
                ):
                    if self.br_cond_edge.check(query_edge, target_edge):
                        self.cand[query_edge].append(target_edge)
        elif self.f[q_i] is not None:
            print("Only Source mapped")
            for t_i, t_j in self.domain.get_domain((q_i, q_j)):
                if t_i == self.f[q_i]:
                    for target_key in self.target.edges_keys((t_i, t_j)):
                        target_edge = (self.f[q_i], t_j, target_key)
                        if (
                            self.target.get_edge_label(target_edge) == self.query.get_edge_label(query_edge) and
                            self.target.node_contains_attributes(t_j, self.query.get_node_attributes(q_j)) and
                            self.target.edge_contains_attributes(target_edge, self.query.get_edge_attributes(query_edge))
                        ):
                            if self.br_cond_node.check(q_j, t_j):
                                self.cand[query_edge].append(target_edge)
        else:
            print("Only Destination mapped")
            for t_i, t_j in self.domain.get_domain((q_i, q_j)):
                if t_j == self.f[q_j]:
                    for target_key in self.target.edges_keys((t_i, t_j)):
                        target_edge = (t_i, self.f[q_j], target_key)
                        if (
                                self.target.get_edge_label(target_edge) == self.query.get_edge_label(query_edge) and
                                self.target.node_contains_attributes(t_i, self.query.get_node_attributes(q_i)) and
                                self.target.edge_contains_attributes(target_edge, self.query.get_edge_attributes(query_edge))
                        ):
                            if self.br_cond_node.check(q_i, t_i):
                                self.cand[query_edge].append(target_edge)

        print("find_candidates", query_edge, self.cand[query_edge])
