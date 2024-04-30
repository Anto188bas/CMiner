from abc import ABC, abstractmethod


# TO DO: if orbits are computed already sorted we don't need to sort them again

class BreakingConditions(ABC):

    def __init__(self, query_graph, mapping_function):
        self.query_graph = query_graph
        self.mapping_function = mapping_function
        self.index = {}
        self.conditions = []

    def index_condition(self):
        """
        Create a dictionary that maps each node to its position in the conditions array.
        :return:
        """
        index = {}
        for i, subarray in enumerate(self.conditions):
            for j, element in enumerate(subarray):
                index[element] = (i, j)
        return index

    def _get_breaking_condition_array(self, elem):
        """
        Return the array of nodes that are breaking conditions for the node q.
        :param q:
        :return: list of nodes
        """
        return self.conditions[self.index[elem][0]]

    def _get_elements_with_smaller_id(self, elem):
        """
        Return the nodes with smaller id than q.
        :param q:
        :return: list of nodes
        """
        return self._get_breaking_condition_array(elem)[:self.index[elem][1]]

    def _get_elements_with_greater_id(self, elem):
        """
        Return the nodes with greater id than q.
        :param q:
        :return: list of nodes
        """
        return self._get_breaking_condition_array(elem)[self.index[elem][1] + 1:]

    def check(self, q, t_element):
        """
        Check if the node q can be mapped to the node t.

        q can be mapped to t if for all nodes q' in the same orbit
        with smaller id than q f(q') != None and f(q') < t
        :param q:
        :param t:
        :return:
        """
        # WRITE BETTER
        elements = self._get_elements_with_smaller_id(q)
        if len(elements) > 0:
            for elem in self._get_elements_with_smaller_id(q):
                if self.mapping_function[elem] is None:
                    continue
                if self.mapping_function[elem] >= t_element:
                    return False
        else:
            elements = self._get_elements_with_greater_id(q)
            for elem in self._get_elements_with_greater_id(q):
                if self.mapping_function[elem] is None:
                    continue
                if self.mapping_function[elem] <= t_element:
                    return False
        return True


class BreakingConditionsNodes(BreakingConditions):

    def __init__(self, query_graph, node_mapping_function):
        super().__init__(query_graph, node_mapping_function)
        self.conditions = [sorted(orbit) for orbit in
                           query_graph.compute_orbits_nodes()]  # Compute orbits already sorted
        self.index = self.index_condition()
        self.breaking_conditions = {}


# class BreakingConditionsEdges(BreakingConditions):
#
#     def __init__(self, query_graph, edge_mapping_function):
#         super().__init__(query_graph, edge_mapping_function)
#         self.conditions = [sorted(orbit) for orbit in query_graph.compute_orbits_edges()]  # Compute orbits already sorted
#         self.index = self.index_condition()
#
class BreakingConditionsEdges:

    def __init__(self, query_graph, edge_mapping_function):
        self.query_graph = query_graph
        self.mapping_function = edge_mapping_function
        self.breaking_conditions = self.init_breaking_conditions()

    def init_breaking_conditions(self):
        breaking_conditions = {}
        # take all src and dest nodes connected by an edge
        edges = set(self.query_graph.edges())

        for src, dest in edges:
            group = {}
            # get all keys of the edges between src and dest
            keys = self.query_graph.edges_keys((src, dest))
            for k in sorted(keys):
                label = self.query_graph.get_edge_label((src, dest, k))
                if label not in group:
                    group[label] = []
                group[label].append(k)

            for label in group:
                breaking_conditions[(src, dest, label)] = group[label]

        return breaking_conditions

    def check(self, e_q, e_t):
        e_src, e_dest, e_key = e_q
        label = self.query_graph.get_edge_label(e_q)
        br_cond_array = self.breaking_conditions.get((e_src, e_dest, label))
        index_key = br_cond_array.index(e_key)
        if index_key is None:
            return True
        br_left_elements = br_cond_array[:index_key]
        for e_left_q in br_left_elements:
            if self.g[e_left_q] is None:
                continue
            if self.g[e_left_q] >= e_t:
                return False
        return True
