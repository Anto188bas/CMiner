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


def print_result(pattern, graphs_where_present, num_pattern):
    print(f"t # {num_pattern}")
    for node in pattern.nodes(data=True):
        print(f"v {node[0]} {' '.join(node[1]['labels'])}")
    for edge in pattern.edges(data=True):
        print(f"e {edge[0]} {edge[1]} {edge[2]['type']}")
    print("x", " ".join([str(i) for i in graphs_where_present]))
    print(f"\nSupport: {len(graphs_where_present)}")
    print("-------------\n")


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
        self.matchers = []
        # at each step we store the frequent patterns found
        self.last_mined_patterns = []

    def mine(self):
        self._read_graphs_from_file()
        self._parse_support()
        self._adapt_target_graphs()
        if self._max_num_nodes > 2:
            # Optimization: pre-compute bit matrices and load them from file
            self._compute_bit_matrices()
            self._init_matcher()
        # define the number of nodes to start mining
        curr_nodes = 1
        # count the number of patterns found
        pattern_count = 0
        while curr_nodes <= self._max_num_nodes:
            # mine patterns
            patterns = self._mine_patterns(curr_nodes)
            if len(patterns) == 0:
                break
            if curr_nodes >= self._min_num_nodes:
                # print results only if the current number of nodes is at least the min number of nodes
                for pattern, graphs_where_present in patterns:
                    print_result(pattern, graphs_where_present, pattern_count)
                    pattern_count += 1
            # increment the number of nodes
            curr_nodes += 1

    def _mine_patterns(self, num_nodes) -> list[tuple[MultiDiGraph, set[int]]]:
        print("\n--------------------------------------")
        print(f"----- MINE PATTERNS WITH {num_nodes} NODES -----")
        print("--------------------------------------\n")

        if num_nodes == 1:
            patterns = self._mine_1node_pattern()
        elif num_nodes == 2:
            patterns = self._mine_2node_pattern()
        else:
            patterns = self._mine_n_node_pattern(num_nodes)
        self.last_mined_patterns = patterns
        return patterns

    def _mine_1node_pattern(self) -> list[tuple[MultiDiGraph, set[int]]]:
        counter = {}
        for i in range(len(self.networks)):
            target = self.networks[i]
            for node in target.nodes(data=True):
                sorted_labels = sorted(node[1]["labels"])
                sorted_labels_str = " ".join(sorted_labels)
                if sorted_labels_str in counter:
                    counter[sorted_labels_str].add(i)
                else:
                    counter[sorted_labels_str] = {i}

        patterns = []
        for sorted_labels_str, graphs_indices in counter.items():
            if len(graphs_indices) >= self.support:
                g = MultiDiGraph()
                g.add_node(1, labels=sorted_labels_str.split(" "))
                patterns.append((g, graphs_indices))
        return patterns

    def _mine_2node_pattern(self) -> list[tuple[MultiDiGraph, set[int]]]:
        counter = {}
        for i in range(len(self.networks)):
            target = self.networks[i]
            for src, dest, attr in target.edges(data=True):
                src_labels = sorted(target.nodes[src]["labels"])
                dest_labels = sorted(target.nodes[dest]["labels"])
                edge_label = attr["type"]
                sorted_labels_str = " ".join(src_labels) + "/" + " ".join(dest_labels) + "/" + edge_label
                if sorted_labels_str in counter:
                    counter[sorted_labels_str].add(i)
                else:
                    counter[sorted_labels_str] = {i}
        patterns = []
        for sorted_labels_str, graphs_indices in counter.items():
            if len(graphs_indices) >= self.support:
                src_labels, dest_labels, edge_label = sorted_labels_str.split("/")
                g = MultiDiGraph()
                g.add_node(1, labels=src_labels.split(" "))
                g.add_node(2, labels=dest_labels.split(" "))
                g.add_edge(1, 2, type=edge_label)
                patterns.append((g, graphs_indices))
        return patterns

    def _mine_n_node_pattern(self, num_nodes) -> list[tuple[MultiDiGraph, set[int]]]:
        """
        Extends existing patterns by adding new nodes and edges based on the support threshold.
        """
        patterns = []

        for pattern, graphs_indices in self.last_mined_patterns:
            out_extensions, in_extensions = {}, {}

            for i in graphs_indices:
                target = self.networks[i]
                matcher = self.matchers[i]
                matcher.match(pattern)
                mappings = matcher.get_solutions()

                for _map in mappings:
                    f = _map.nodes_mapping()
                    target_nodes = set(f.values())

                    for node_q, node_t in f.items():
                        self._extend_patterns(target, node_q, node_t, target_nodes, out_extensions, in_extensions, i)

            # Create patterns with outgoing and incoming extensions
            patterns.extend(self._create_patterns(pattern, out_extensions, num_nodes, True))
            patterns.extend(self._create_patterns(pattern, in_extensions, num_nodes, False))

        self._remove_redundant_patterns(patterns)

        return patterns

    def _remove_redundant_patterns(self, patterns):
        codes = set()
        for g, _ in patterns:
            code = g.code()
            if code not in codes:
                codes.add(code)
            else:
                patterns.remove((g, _))

    def _extend_patterns(self, target, node_q, node_t, target_nodes, out_extensions, in_extensions, i):
        """
        Finds all unmapped neighbors for each node in the mapping and adds them to extensions.
        """
        for neigh in set(target.successors(node_t)).difference(target_nodes):
            self._add_extension(out_extensions, node_q, neigh, target, node_t, i, True)
        for neigh in set(target.predecessors(node_t)).difference(target_nodes):
            self._add_extension(in_extensions, node_q, neigh, target, node_t, i, False)

    def _add_extension(self, extensions, node_q, neigh, target, node_t, graph_index, is_outgoing):
        """
        Adds an extension to the appropriate dictionary based on whether the edge is outgoing or incoming.
        """
        edge_keys = target.edges_keys((node_t, neigh)) if is_outgoing else target.edges_keys((neigh, node_t))

        for edge_key in edge_keys:
            edge_label = target.get_edge_label((node_t, neigh, edge_key)) if is_outgoing else target.get_edge_label(
                (neigh, node_t, edge_key))
            target_node_labels_code = " ".join(target.get_node_labels(neigh))
            extension_code = (node_q, target_node_labels_code, edge_label)

            if extension_code not in extensions:
                extensions[extension_code] = set()
            extensions[extension_code].add(graph_index)

    def _create_patterns(self, pattern, extensions, num_nodes, is_outgoing):
        """
        Creates new patterns from the extensions that meet the support threshold.
        """
        patterns = []

        for extension_code, graphs in extensions.items():

            if len(graphs) >= self.support:
                query_node_id, target_node_labels_code, edge_label = extension_code
                target_node_labels = target_node_labels_code.split(" ")
                p = MultiDiGraph(pattern)
                p.add_node(num_nodes, labels=target_node_labels)
                if is_outgoing:
                    p.add_edge(query_node_id, num_nodes, type=edge_label)
                else:
                    p.add_edge(num_nodes, query_node_id, type=edge_label)
                patterns.append((p, graphs))

        return patterns

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

    # def _mine_n_node_pattern(self, num_nodes) -> list[tuple[MultiDiGraph, set[int]]]:
    #     # Extend every pattern with n - 1 nodes (stored in self.last_mined_patterns)
    #     patterns = []
    #     """
    #         P1   P2   .. Pk
    #     G1  maps maps .. maps
    #     G2  maps maps .. maps
    #     ..  ...  ...  ..  ..
    #     Gn  maps maps .. maps
    #
    #     Cycle by columns
    #     Find all mapping for each pattern in each graph of the column
    #     Find all neighbors of the mapped nodes in the target graph (not already mapped)
    #     """
    #     for pattern, graphs_indices in self.last_mined_patterns:
    #         extensions = {}
    #         for i in graphs_indices:
    #             target = self.networks[i]
    #             matcher = self.matchers[i]
    #             matcher.match(pattern)
    #             mappings = matcher.get_solutions()
    #             for _map in mappings:
    #                 f = _map.nodes_mapping()
    #                 target_nodes = f.values()
    #                 for node_q, node_t in f.items():
    #                     for neigh in set(self.networks[i].successors(node_t)).difference(target_nodes):
    #                         edge_keys = target.edges_keys((node_t, neigh))
    #                         for edge_key in edge_keys:
    #                             edge_label = target.get_edge_label((node_t, neigh, edge_key))
    #                             target_node_labels_code = " ".join(target.get_node_labels(neigh))
    #                             extension_code = (node_q, target_node_labels_code, edge_label)
    #                             if extension_code not in extensions:
    #                                 extensions[extension_code] = set()
    #                                 extensions[extension_code].add(i)
    #                             else:
    #                                 extensions[extension_code].add(i)
    #         # keep only the extensions that are present in at least self.support graphs
    #         for extension_code, _graphs in extensions.items():
    #             if len(_graphs) >= self.support:
    #                 query_node_id, target_node_labels_code, edge_label = extension_code
    #                 target_node_labels = target_node_labels_code.split(" ")
    #                 pattern.add_node(num_nodes, labels=target_node_labels)
    #                 pattern.add_edge(query_node_id, num_nodes, type=edge_label)
    #                 patterns.append((pattern, _graphs))
    #     return patterns
