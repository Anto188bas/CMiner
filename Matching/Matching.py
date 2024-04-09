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
        forceBack = False
        i = 0
        q_i, q_j, q_key = query_edge = self.ordering.get(i)
        self._find_candidates(query_edge)
        while i >= 0:
            print("\n\nQuery edge: ", query_edge)
            print("Candidates: ", self.cand[query_edge])
            print("Index candidates: ", self.cand_index[query_edge])
            if forceBack or self.cand_index[query_edge] >= len(self.cand[query_edge]):
                print("BACKTRACKING")
                forceBack = False
                if self.g[query_edge] is None:
                    i -= 1
                if i < 0:
                    # no more solutions
                    break
                q_i, q_j, q_key = query_edge = self.ordering.get(i)

                # backtraking
                # RESET THE MAPPING OF THE QUERY EDGE
                print(self.g)
                self.g[query_edge] = None
                print("g[", query_edge, "] = None")
                print(self.g)
                # RESET THE MAPPING OF THE QUERY NODES
                # reset f[q_i] if there are no mapped edges in the query that have q_i as source/destination
                if all(self.g[q_e] is None for q_e in self.query.get_all_edges() if q_e[0] == q_i or q_e[1] == q_i):
                    print("Reset f[", q_i, "] = None")
                    self.f[q_i] = None
                # reset f[q_j] if there are no mapped edges in the query that have q_j as source/destination
                if all(self.g[q_e] is None for q_e in self.query.get_all_edges() if q_e[0] == q_j or q_e[1] == q_j):
                    print("Reset f[", q_j, "] = None")
                    self.f[q_j] = None
                # take the previous edge in the ordering
                # i -= 1
                # if i < 0:
                #     # no more solutions
                #     break
                # q_i, q_j, q_key = query_edge = self.ordering.get(i)
                self.cand_index[query_edge] += 1  #################################

            else:
                # CHECK IF THE NEXT CANDIDATE TARGET EDGE IS COMPATIBLE WITH THE QUERY EDGE
                # extract the target edge
                t_i, t_j, key = target_edge = self.cand[query_edge][self.cand_index[query_edge]]
                print("Target edge: ", target_edge)
                # check if the target edge has not been already mapped to another query edge
                if all(self.g[q_e] != target_edge for q_e in self.query.get_all_edges() if self.g[q_e] is not None):
                    # if the mapping for the source is not already set, set it
                    if self.f[q_i] is None:
                        print("Set f[", q_i, "] = ", t_i)
                        self.f[q_i] = t_i
                    # if the mapping for the destination is not already set, set it
                    if self.f[q_j] is None:
                        print("Set f[", q_j, "] = ", t_j)
                        self.f[q_j] = t_j
                    # set the mapping for the query edge
                    print("Set g[", query_edge, "] = ", target_edge)
                    self.g[query_edge] = target_edge
                    # if the edge is the last in the ordering, the solution is found
                    if i == len(self.query.get_all_edges()) - 1:
                        print("NEW SOLUTION FOUND")
                        forceBack = True
                        # SOLUTION FOUND
                        # save the mapping
                        self.occurrences.append({
                            "f": self.f.copy(),
                            "g": self.g.copy()
                        })
                        # shift the index to the next candidate
                        # self.cand_index[query_edge] += 1
                    else:
                        print("NEXT QUERY EDGE")
                        # shift the index to the next query edge in the ordering
                        i += 1
                        # get the next query edge
                        q_i, q_j, q_key = query_edge = self.ordering.get(i)
                        # find the candidates for the next query edge
                        self._find_candidates(query_edge)
                        # reset the index for the candidates of the next query edge
                        self.cand_index[query_edge] = 0
                else:
                    print("Already mapped")
                    # shift the index to the next candidate
                    self.cand_index[query_edge] += 1
        print("SOLUTIONS FOUND: ", len(self.occurrences))
        for occ in self.occurrences:
            print("\n\nSolution:")
            print("f: ", occ["f"])
            print("g: ", occ["g"])

    def match_no_print(self):
        self.ordering.compute()
        # used to force backtracking when a solution is found
        force_backtracking = False
        # index of the current query edge in the ordering
        i = 0
        # get the first query edge
        q_i, q_j, q_key = query_edge = self.ordering.get(i)
        # find the candidates for the first query edge
        self._find_candidates(query_edge)
        # loop until all the query edges have been mapped
        while i >= 0:
            # if the index of the candidates is greater than the number of candidates
            # or the backtracking is forced
            if force_backtracking or self.cand_index[query_edge] >= len(self.cand[query_edge]):
                force_backtracking = False
                if self.g[query_edge] is None:
                    i -= 1
                if i < 0:
                    # no more solutions
                    break
                q_i, q_j, q_key = query_edge = self.ordering.get(i)

                # backtraking
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
                        force_backtracking = True
                        # SOLUTION FOUND
                        # save the mapping
                        self.occurrences.append({
                            "f": self.f.copy(),
                            "g": self.g.copy()
                        })
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
        print("Find candidates for: ", query_edge)
        self.cand[query_edge] = []
        if self.f[q_i] is None and self.f[q_j] is None:
            print("1")
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
            print("2 AAAAAAAAAAAAAAAAAAAAAAAA")
            for target_key in self.target.edges_keys((self.f[q_i], self.f[q_j])):
                target_edge = (self.f[q_i], self.f[q_j], target_key)
                print("Target edge: ", target_edge)
                if (
                        self.target.get_edge_label(target_edge) == self.query.get_edge_label(query_edge) and
                        self.target.edge_contains_attributes(target_edge, self.query.get_edge_attributes(query_edge))
                ):
                    if self.br_cond_edge.check(query_edge, target_edge):
                        self.cand[query_edge].append(target_edge)
        elif self.f[q_i] is not None:
            print("3")
            for t_i, t_j in self.domain.get_domain((q_i, q_j)):
                if t_i == self.f[q_i]:
                    for target_key in self.target.edges_keys((t_i, t_j)):
                        target_edge = (t_i, t_j, target_key)
                        if (
                                self.target.get_edge_label(target_edge) == self.query.get_edge_label(query_edge) and
                                self.target.node_contains_attributes(t_j, self.query.get_node_attributes(q_j)) and
                                self.target.edge_contains_attributes(target_edge,
                                                                     self.query.get_edge_attributes(query_edge))
                        ):
                            if self.br_cond_node.check(q_j, t_j):
                                self.cand[query_edge].append(target_edge)
        else:
            print("4")
            print(self.domain.get_domain((q_i, q_j)))
            for t_i, t_j in self.domain.get_domain((q_i, q_j)):
                print((t_i, t_j))
                print("aaa", t_j, self.f[q_j])
                if t_j == self.f[q_j]:
                    print("HERE")
                    for target_key in self.target.edges_keys((t_i, t_j)):
                        target_edge = (t_i, t_j, target_key)

                        if (
                                self.target.get_edge_label(target_edge) == self.query.get_edge_label(query_edge) and
                                self.target.node_contains_attributes(t_i, self.query.get_node_attributes(q_i)) and
                                self.target.edge_contains_attributes(target_edge,
                                                                     self.query.get_edge_attributes(query_edge))
                        ):
                            if self.br_cond_node.check(q_i, t_i):
                                self.cand[query_edge].append(target_edge)
