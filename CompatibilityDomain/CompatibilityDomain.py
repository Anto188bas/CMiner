class CompatibilityDomain:

    def __init__(self, qbm, tbm):
        self.qbm = qbm
        self.tbm = tbm
        self.domain = {}

    def _check_conditions(self, query_edge, target_edge):
        q_graph = self.qbm.get_graph()
        t_graph = self.tbm.get_graph()
        q_i = query_edge[0]
        q_j = query_edge[1]
        t_i = target_edge[0]
        t_j = target_edge[1]
        labels = t_graph.get_all_edge_labels()
        for label in labels:
            if q_graph.t_out_deg(q_i, label) > t_graph.t_out_deg(t_i, label):
                return False
            if q_graph.t_in_deg(q_i, label) > t_graph.t_in_deg(t_i, label):
                return False
            if q_graph.t_out_deg(q_j, label) > t_graph.t_out_deg(t_j, label):
                return False
            if q_graph.t_in_deg(q_j, label) > t_graph.t_in_deg(t_j, label):
                return False
        return True

    def compute(self):
        candidates = self.qbm.find_candidates(self.tbm)
        qbm_indices = self.qbm.get_matrix_indices()
        tbm_indices = self.tbm.get_matrix_indices()
        for i in range(0, len(qbm_indices), 2):
            self.domain[qbm_indices[i]] = []
        for candidate in candidates:
            query_edge_index = candidate[0]
            target_edge_index = candidate[1]
            if query_edge_index & 1 == 0:
                query_edge = qbm_indices[query_edge_index]
                target_edge = tbm_indices[target_edge_index]
            else:
                query_edge = qbm_indices[query_edge_index - 1]
                target_edge = (tbm_indices[target_edge_index][1], tbm_indices[target_edge_index][0])

            if self._check_conditions(query_edge, target_edge):
                self.domain[query_edge].append(target_edge)


        return self.domain