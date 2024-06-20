import os
from src.NetworkX.NetworkConfigurator import NetworkConfigurator
from src.NetworkX.NetworksLoading import NetworksLoading
from src.CMiner.MultiGraphMatch import MultiGraphMatch
from src.Graph.Graph import MultiDiGraph
from src.CMiner.BitMatrix import TargetBitMatrixOptimized, BitMatrixStrategy2
import time

matching_total_time = 0


def get_file_details(file_path):
    path = os.path.dirname(file_path)
    filename_with_extension = os.path.basename(file_path)
    filename, extension = os.path.splitext(filename_with_extension)
    return path, filename, extension


# TO-DO: fix generation of 2 node patterns
class CMiner:

    def __init__(self,
                 db_file_name,
                 min_num_nodes=1,
                 max_num_nodes=float('inf'),
                 support=0.5,
                 start_pattern=None
                 ):
        self.db_file_name = db_file_name
        self._min_num_nodes = min_num_nodes
        self._max_num_nodes = max_num_nodes
        self.support = support
        self._start_pattern = start_pattern
        self.networks = []
        self.networks_names = []
        self.target_bit_matrices = []
        self.frequent_patterns = {}
        self.matchers = []

    def mine(self):
        self._read_graphs_from_file()
        self._parse_support()
        self._adapt_target_graphs()
        self._compute_bit_matrices()
        self._init_matcher()
        # define the number of nodes to start mining
        curr_nodes = self._min_num_nodes
        while curr_nodes <= self._max_num_nodes:
            self._mine_patterns(curr_nodes)
            curr_nodes += 1

    def _mine_patterns(self, num_nodes):
        global matching_total_time
        start = time.time()
        if num_nodes == 1:
            start = time.time()
            self._mine_1node_pattern()
            end = time.time()
            print(f"Mine {num_nodes} nodes patterns - Execution time: {end - start} seconds")
            return
        elif num_nodes == 2:
            patterns = self._generate_2node_pattern()
        else:
            patterns = []
        for graph, graph_indices in patterns:
            result = self._graphs_where_pattern_is_present(graph, graph_indices)
            if len(result) < self.support:
                continue
            if num_nodes not in self.frequent_patterns:
                self.frequent_patterns[num_nodes] = []
            self.frequent_patterns[num_nodes].append(
                (graph, [self.networks_names[i] for i, _ in result], [s for _, s in result]))
        end = time.time()
        print(f"Mine {num_nodes} nodes patterns - Execution time: {end - start} seconds")
        print(f"Matching total time: {matching_total_time} seconds")
        matching_total_time = 0

    def _graphs_where_pattern_is_present(self, graph, graphs_indices):
        """
        Check if the patterns are frequent in the db.
        A pattern is frequent if it is present in at least
        self.support graphs in the db.
        """
        global matching_total_time
        graphs_where_present = []
        for i in graphs_indices:
            matcher = self.matchers[i]
            start = time.time()
            matcher.match(graph)
            end = time.time()
            matching_total_time += (end - start)
            solutions = matcher.get_solutions()
            if len(solutions) > 0:
                graphs_where_present.append((i, solutions))
        return graphs_where_present

    def _generate_2node_pattern(self):
        simple_patterns = self._frequent_simple_pattern()
        L_fist_patterns = simple_patterns["L_first"]
        L_second_patterns = simple_patterns["L_second"]
        T_in_patterns = simple_patterns["T_in"]
        T_out_patterns = simple_patterns["T_out"]

        # generate all possible 2-node patterns combining the simple patterns
        # that are frequent in the db. This method reduce the number of patterns

        possible_patterns = []

        for node_labels_1, graph_indices_1 in L_fist_patterns:
            for node_labels_2, graph_indices_2 in L_second_patterns:
                # Check if the tho patterns merged can be frequent in the db.
                # If the number of graphs containing the two nodes is less than the support
                # then the pattern is not frequent
                common_graphs_indices = set(graph_indices_1).intersection(graph_indices_1)
                if len(common_graphs_indices) < self.support:
                    continue
                # generate graph of this type: node_labels_1 -> node_labels_2
                for edge_labels, graph_indices in T_out_patterns:
                    common_graphs_indices_with_pattern = set(common_graphs_indices).intersection(graph_indices)
                    if len(common_graphs_indices_with_pattern) < self.support:
                        continue

                    # FIX, I KNOW FOR SURE THAT THIS IS NOT 100% CORRECT
                    for edge_label in edge_labels:
                        g = MultiDiGraph()
                        g.add_node(1, labels=node_labels_1)
                        g.add_node(2, labels=node_labels_2)
                        g.add_edge(1, 2, type=edge_label)
                        possible_patterns.append((g, common_graphs_indices))
                # generate graph of this type: node_labels_1 <- node_labels_2
                for edge_labels, graph_indices in T_in_patterns:
                    common_graphs_indices_with_pattern = set(common_graphs_indices).intersection(graph_indices)
                    if len(common_graphs_indices_with_pattern) < self.support:
                        continue
                    # FIX, I KNOW FOR SURE THAT THIS IS NOT 100% CORRECT
                    for edge_label in edge_labels:
                        g = MultiDiGraph()
                        g.add_node(1, labels=node_labels_1)
                        g.add_node(2, labels=node_labels_2)
                        g.add_edge(2, 1, type=edge_label)
                        possible_patterns.append((g, common_graphs_indices))

        return possible_patterns

    def _mine_1node_pattern(self):
        """
        All patterns are mined directly from the bit matrices
        """

        simple_patterns = self._frequent_simple_pattern()
        L_fist_patterns = simple_patterns["L_first"]
        L_second_patterns = simple_patterns["L_second"]
        # there can exist patterns redundant in L_first and L_second
        # if some pattern is redundant we merge the list of graphs id containing the pattern
        node_patterns = []
        # add all patterns from L_first
        for labels, graphs_indexes in L_fist_patterns:
            node_patterns.append((labels, graphs_indexes))
        # add all patterns from L_second
        for labels, graphs_indexes in L_second_patterns:
            found = False
            # check if the pattern is already in the list
            for i, (l, gi) in enumerate(node_patterns):
                # LABELS ARE SORTED SO THIS SHOULD NOT BE A PROBLEM. !!!!!!!!!!!!!!!!!!!!!!!!!!
                if labels == l:
                    node_patterns[i] = (l, gi + graphs_indexes)
                    found = True
                    break
            if not found:
                node_patterns.append((labels, graphs_indexes))

        self.frequent_patterns[1] = []
        for labels, graphs_indexes in node_patterns:
            g = MultiDiGraph()
            g.add_node(1, labels=labels)
            self.frequent_patterns[1].append((g, [self.networks_names[i] for i in graphs_indexes]))

    def print_results(self):
        for num_nodes, patterns in self.frequent_patterns.items():
            print(f"Frequent patterns with {num_nodes} nodes: {len(patterns)}")
            if num_nodes > 1:
                for graph, db_names, mappings in patterns:
                    print("-------------")
                    print(graph.nodes(data=True))
                    print(graph.edges(data=True))
                    print(f"Inside {len(db_names)} graphs")
                    print(db_names)
                    print("Example of mapping")
                    print(mappings[0][0])
            else:
                for graph, db_names in patterns:
                    print("-------------")
                    print(graph.nodes(data=True))
                    print(graph.edges(data=True))
                    print(f"Inside {len(db_names)} graphs")
                    print(db_names)

    def _frequent_simple_pattern(self):
        """
        Find frequent simple patterns in the db.
        A simple pattern is a single node or an edge.

        It returns a dictionary with the following structure:
        {
            "L_first": : [(list of node labels, list of graph indexes), ...],
            "L_second": : [(list of node labels, list of graph indexes), ...],
            "T_in": : [(list of edge labels, list of graph indexes), ...],
            "T_out": : [(list of edge labels, list of graph indexes), ...]
        }
        each element of the list represents a node with all of its labels and the
        list of graph indexes that contain that node.
        """

        def update_patterns(value_dict, pattern_dict, graph_idx):
            """
            Update the pattern dictionary with values and their corresponding graph indices.

            Args:
                value_dict (dict): Dictionary containing distinct values.
                pattern_dict (dict): Dictionary to update with graph indices.
                graph_idx (int): Index of the current graph.
            """
            for value in value_dict:
                if value in pattern_dict:
                    pattern_dict[value].append(graph_idx)
                else:
                    pattern_dict[value] = [graph_idx]

        def process_patterns(pattern_dict, all_labels):
            """
            Process the pattern dictionary to extract labels and their occurrences in graphs.

            Args:
                pattern_dict (dict): Dictionary containing patterns and graph indices.
                all_labels (list): List of all possible labels (node or edge).

            Returns:
                list: List of tuples containing labels and their support count.
            """
            patterns = []
            for bitmap, graphs in pattern_dict.items():
                if len(graphs) >= self.support:
                    labels = [all_labels[i] for i in range(len(bitmap)) if bitmap[i] == "1"]
                    patterns.append((labels, graphs))
            return patterns

        L_first = {}
        L_second = {}
        T_in = {}
        T_out = {}

        # For all networks
        for i, tbm in enumerate(self.target_bit_matrices):
            # Update patterns for L_first, L_second, T_in, and T_out
            update_patterns(tbm.L_first_distinct_values(), L_first, i)
            update_patterns(tbm.L_second_distinct_values(), L_second, i)
            update_patterns(tbm.T_in_distinct_values(), T_in, i)
            update_patterns(tbm.T_out_distinct_values(), T_out, i)

        # List of all node and edge labels in the database
        # All graphs in the db have the same labels so we can take the labels from the first graph
        all_node_labels = self.networks[0].get_all_node_labels()
        all_edge_labels = self.networks[0].get_all_edge_labels()

        # Construct the frequent patterns dictionary
        frequent_patterns = {
            "L_first": process_patterns(L_first, all_node_labels),
            "L_second": process_patterns(L_second, all_node_labels),
            "T_in": process_patterns(T_in, all_edge_labels),
            "T_out": process_patterns(T_out, all_edge_labels),
        }

        return frequent_patterns

    def _adapt_target_graphs(self):
        """
        The BitMatrix for each target is created based on the
        labels of nodes and edges in the target graph. In the
        DB of graphs, there could exist graphs without all the
        label. So we need to adapt the target graphs so that
        they have the same labels set.
        :return:
        """
        start_time = time.time()
        # get all labels in the db
        all_node_labels = set()
        all_edge_labels = set()
        for g in self.networks:
            all_node_labels = all_node_labels.union(g.get_all_node_labels())
            all_edge_labels = all_edge_labels.union(g.get_all_edge_labels())

        # for each graph in the db add the missing labels
        for g in self.networks:
            missing_node_labels = all_node_labels.difference(g.get_all_node_labels())
            missing_edge_labels = all_edge_labels.difference(g.get_all_edge_labels())
            g.add_node('dummy', labels=missing_node_labels)
            for label in missing_edge_labels:
                g.add_edge('dummy', 'dummy', type=label)
        end_time = time.time()
        print(f"Adapt target graphs - Execution time: {end_time - start_time} seconds")

    def _read_graphs_from_file(self):
        start_time = time.time()
        type_file = "data"
        configurator = NetworkConfigurator(type_file)
        for name, network in NetworksLoading(type_file, configurator.config).Networks.items():
            self.networks.append(MultiDiGraph(network))
            self.networks_names.append(name)
        end_time = time.time()
        print(f"Read graphs - Execution time: {end_time - start_time} seconds")

    def _compute_bit_matrices(self):
        start_time = time.time()
        for target in self.networks:
            tbm = TargetBitMatrixOptimized(target, BitMatrixStrategy2())
            tbm.compute()
            self.target_bit_matrices.append(tbm)
        end_time = time.time()
        print(f"Compute bit matrices - Execution time: {end_time - start_time} seconds")

    def _parse_support(self):
        """
        If the support is > 1, then the user want common
        graphs that are present in a certain amount of
        db graphs.
        If the support is <= 1 then the user want common
        graphs that are present in a certain percentage of
        df graphs.
        """
        if self.support <= 1:
            db_len = len(self.networks)
            self.support = int(self.support * db_len)

    def _init_matcher(self):
        start_time = time.time()
        for i in range(len(self.networks)):
            matcher = MultiGraphMatch(self.networks[i], target_bit_matrix=self.target_bit_matrices[i])
            self.matchers.append(matcher)
        end_time = time.time()
        print(f"Init matchers - Execution time: {end_time - start_time} seconds")
