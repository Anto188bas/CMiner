import os
from src.NetworkX.NetworkConfigurator import NetworkConfigurator
from src.NetworkX.NetworksLoading import NetworksLoading
from src.CMiner.MultiGraphMatch import MultiGraphMatch
from src.Graph.Graph import MultiDiGraph
from src.CMiner.BitMatrix import TargetBitMatrixOptimized, BitMatrixStrategy2
import copy
import time


def debug_message(*args, **kwargs):
    if False:
        debug_message(*args, **kwargs)
        
matching_total_time = 0
count_pattern = 0


def get_file_details(file_path):
    path = os.path.dirname(file_path)
    filename_with_extension = os.path.basename(file_path)
    filename, extension = os.path.splitext(filename_with_extension)
    return path, filename, extension


class DBGraph(MultiDiGraph):

    def __init__(self, graph, name):
        super().__init__(graph)
        self.name = name
        self.matcher = None

    def subgraph(self, nodes):
        """
        Return a subgraph of the graph containing only the nodes passed as argument.
        """
        subgraph = MultiDiGraph()
        for node in nodes:
            subgraph.add_node(node, labels=self.nodes[node]['labels'])
        for src, dest, key in self.edges(keys=True):
            if src in nodes and dest in nodes:
                subgraph.add_edge(src, dest, key=key, type=self.get_edge_label((src, dest, key)))
        return subgraph

    def init_matcher(self):
        bit_matrix = TargetBitMatrixOptimized(self, BitMatrixStrategy2())
        bit_matrix.compute()
        self.matcher = MultiGraphMatch(self, target_bit_matrix=bit_matrix)

    def localize(self, pattern):
        self.matcher.match(pattern)
        return self.matcher.get_solutions()

    def get_name(self):
        return self.name


class ExtensionManager:

    def __init__(self, support):
        self.support = support
        self.out_extensions = {}
        self.in_extensions = {}
        self.out_extensions_node_ids = {}
        self.in_extensions_node_ids = {}

    def add_extension(self, pattern_node_id, target_src_node_id, target_dst_node_id, is_outgoing, db_graph):
        target_edge_labels_code = " ".join(db_graph.get_edge_labels(target_src_node_id, target_dst_node_id))
        if is_outgoing:
            self._add_extension_helper(self.out_extensions, self.out_extensions_node_ids, pattern_node_id,
                                       target_dst_node_id, target_edge_labels_code, db_graph)
        else:
            self._add_extension_helper(self.in_extensions, self.in_extensions_node_ids, pattern_node_id,
                                       target_src_node_id, target_edge_labels_code, db_graph)

    def _add_extension_helper(self, extensions, extensions_node_ids, pattern_node_id, target_node_id,
                              target_edge_labels_code, db_graph):
        target_node_labels_code = " ".join(db_graph.get_node_labels(target_node_id))
        extension_code = (pattern_node_id, target_node_labels_code, target_edge_labels_code)
        if extension_code not in extensions:
            extensions[extension_code] = set()
        extensions[extension_code].add(db_graph)

        # 🚫
        if extension_code not in extensions_node_ids:
            extensions_node_ids[extension_code] = {}
        if db_graph not in extensions_node_ids[extension_code]:
            extensions_node_ids[extension_code][db_graph] = set()
        extensions_node_ids[extension_code][db_graph].add(target_node_id)
        # 🚫

    def get_extensions(self) -> list['Extension']:
        out_extensions = self._filter_extensions(self.out_extensions, self.out_extensions_node_ids, True)
        in_extensions = self._filter_extensions(self.in_extensions, self.in_extensions_node_ids, False)
        return out_extensions + in_extensions

    def _filter_extensions(self, extensions, extensions_node_ids, is_outgoing) -> list['Extension']:
        """
        Return a list of Extensions that are frequent
        :param extensions:
        :return:
        """
        filtered_extensions = []
        for (pattern_node_id, target_node_labels, target_edge_labels), db_graphs in extensions.items():
            if len(db_graphs) >= self.support:
                # 🚫
                extensions_node_id = extensions_node_ids[(pattern_node_id, target_node_labels, target_edge_labels)]
                # 🚫
                target_node_labels = target_node_labels.split(" ")
                target_edge_labels = target_edge_labels.split(" ")
                ext = Extension(pattern_node_id, target_node_labels, target_edge_labels, is_outgoing, db_graphs,
                                extensions_node_id)
                filtered_extensions.append(ext)
        return filtered_extensions


class Extension:
    def __init__(self, pattern_node_id, target_node_labels, target_edge_labels, is_outgoing, db_graphs,
                 extension_node_ids):
        self.pattern_node_id = pattern_node_id
        self.target_node_labels = target_node_labels
        self.target_edge_labels = target_edge_labels
        self.is_outgoing = is_outgoing
        self.db_graphs = db_graphs
        # 🚫
        self.extension_node_ids = extension_node_ids

    def __str__(self):
        if self.is_outgoing:
            output = f"({self.pattern_node_id} -- {self.target_edge_labels} --> {self.target_node_labels})"
        else:
            output = f"({self.pattern_node_id} <-- {self.target_edge_labels} -- {self.target_node_labels})"
        return output


class Pattern(MultiDiGraph):

    def __init__(self, graph, db_graphs, mappings=None):
        super().__init__(graph)
        self.db_graphs = db_graphs
        self.mappings = mappings

    def __copy__(self):
        # Copia superficiale dell'array db_graphs (nuovo array con gli stessi oggetti)
        db_graphs_copy = copy.copy(self.db_graphs)

        # Copia profonda del dizionario mappings (nuovo dizionario con nuovi array come valori)
        start_time = time.time()
        debug_message("        - DEEPCOPY")
        mappings_copy = {k: copy.deepcopy(v) for k, v in self.mappings.items()}
        end_time = time.time()
        debug_message(f"        - DEEPCOPY: {end_time - start_time} seconds")

        # Crea un nuovo oggetto Pattern con le copie
        return Pattern(self, db_graphs_copy, mappings_copy)

    def extend(self, support) -> list['Pattern']:
        start_time = time.time()
        extension_manager = ExtensionManager(support)
        # set of codes of the patterns
        codes = set()
        # list containing pattern generated
        patterns = []
        # search all possible extension
        for ext in self._find_extensions(extension_manager):
            new_patter = self._apply_extension(ext)
            new_patter_code = new_patter.code()
            if new_patter_code not in codes:
                patterns.append(new_patter)
                codes.add(new_patter_code)
                new_patter._update_mappings(ext, max(self.nodes()) + 1)
            else:
                debug_message("Pattern already found")

        # for p in patterns:
        #     p.find_cycles(support)

        patterns = remove_redundant_patterns(patterns)
        end_time = time.time()
        debug_message(f"- EXTEND PATTERN: {end_time - start_time} seconds")
        return patterns

    def find_cycles(self, support) -> list['Pattern']:
        # debug_message("Find cycles")
        # debug_message(self)
        # debug_message("Find cycles for pattern")
        # debug_message(self)
        candidate_edges_to_extend_count = {}
        for g in self.db_graphs:
            for _map in self.mappings[g]:
                # all nodes in g that are mapped with the pattern
                mapped_nodes = set(_map.target_nodes())
                # all edges in g that are mapped with the pattern
                mapped_edges = set(_map.target_edges())
                # projection of pattern in g
                # containing all edges between nodes that are mapped
                projection = g.subgraph(mapped_nodes)
                # the edge candidates to extend are all edges in the projection
                # that are not already mapped
                candidate_target_edges_to_extend = set(projection.edges(keys=True)).difference(mapped_edges)
                # debug_message("Candidate target edges to extend", candidate_target_edges_to_extend)
                for target_edge in candidate_target_edges_to_extend:
                    # get the edge labels of the target edge
                    target_edge_label = g.get_edge_label(target_edge)
                    target_edge_src = target_edge[0]
                    target_edge_dst = target_edge[1]
                    # get the pattern node that is mapped to the source of the target edge
                    for n in self.nodes():
                        if _map.get_node_mapping(n) == target_edge_src:
                            pattern_node_src = n
                            break
                    # get the pattern node that is mapped to the destination of the target edge
                    for n in self.nodes():
                        if _map.get_node_mapping(n) == target_edge_dst:
                            pattern_node_dst = n
                            break
                    ext = (pattern_node_src, pattern_node_dst, target_edge_label)
                    if ext not in candidate_edges_to_extend_count:
                        candidate_edges_to_extend_count[ext] = set()
                    # add the graph to the set of graphs that contains the edge
                    candidate_edges_to_extend_count[ext].add(g)
        frequent_edges = {edge: graphs for edge, graphs in candidate_edges_to_extend_count.items() if
                          len(graphs) >= support}
        # The pattern we try to extend adding a cycle, for a specific graph,
        # can contain or not contain the edge that we want to add.
        # If the edge is contained in the pattern for a specific graph,
        # then the extension is made and a cycle is created.
        # If the edge is not contained in the pattern for a specific graph,
        # then the extension is not made and the pattern remains the same.

        # debug_message("ccc", frequent_edges.items())
        for edge, graphs in frequent_edges.items():
            self.add_edge(edge[0], edge[1], type=edge[2])
            self.db_graphs = graphs
        # debug_message("New pattern")
        # debug_message(self)

    def _apply_extension(self, extension) -> 'Pattern':
        start_time = time.time()
        debug_message("    - APPLY EXTENSION")
        new_pattern = self.__copy__()
        new_node_id = max(self.nodes()) + 1
        new_pattern.add_node(new_node_id, labels=extension.target_node_labels)
        for edge_label in extension.target_edge_labels:
            if extension.is_outgoing:
                new_pattern.add_edge(extension.pattern_node_id, new_node_id, type=edge_label)

            else:
                new_pattern.add_edge(new_node_id, extension.pattern_node_id, type=edge_label)
        end_time = time.time()
        debug_message(f"    - APPLY EXTENSION: {end_time - start_time} seconds")
        return new_pattern

    def _update_mappings(self, extension, new_node_id):
        start_time = time.time()
        debug_message("    - UPDATE MAPPINGS")

        new_mappings = {}
        for g in extension.db_graphs:
            new_mappings[g] = self.get_mappings(g)
        self.mappings = new_mappings
        self.db_graphs = extension.db_graphs

        # node of the pattern from which the extension is applied
        node_p = extension.pattern_node_id

        # update mappings
        for g in self.db_graphs:
            tmp_mappings = []
            for target_node_id in extension.extension_node_ids[g]:
                debug_message("        - MAPPINGS", len(self.mappings[g]))
                for _map in self.mappings[g]:
                    # mapping of the node_p in the DB graph
                    node_db = _map.get_node_mapping(node_p)

                    if extension.is_outgoing:
                        if target_node_id in set(g.successors(node_db)):
                            new_map = _map.copy()
                            new_map.add_node_mapping(new_node_id, target_node_id)
                            key_p = 0
                            for label_p in self.get_edge_labels(node_p, new_node_id):
                                for key_db in g.edges_keys((node_db, target_node_id)):
                                    edge_db = (node_db, target_node_id, key_db)
                                    label_db = g.get_edge_label(edge_db)
                                    if label_p == label_db and not new_map.is_edge_mapped(edge_db):
                                        new_map.add_edge_mapping((node_p, new_node_id, key_p), edge_db)
                                        key_p += 1
                            tmp_mappings.append(new_map)
                    else:
                        if target_node_id in set(g.predecessors(node_db)):
                            new_map = _map.copy()
                            new_map.add_node_mapping(new_node_id, target_node_id)
                            key_p = 0
                            for label_p in self.get_edge_labels(new_node_id, node_p):
                                for key_db in g.edges_keys((target_node_id, node_db)):
                                    edge_db = (target_node_id, node_db, key_db)
                                    label_db = g.get_edge_label(edge_db)
                                    if label_p == label_db and not new_map.is_edge_mapped(edge_db):
                                        new_map.add_edge_mapping((new_node_id, node_p, key_p), edge_db)
                                        key_p += 1
                            tmp_mappings.append(new_map)
            self.mappings[g] = tmp_mappings
        end_time = time.time()
        debug_message(f"    - UPDATE MAPPINGS: {end_time - start_time} seconds")

    # def _update_mappings(self, extension, new_node_id):
    #     start_time = time.time()
    #     debug_message("UPDATE MAPPINGS")
    #
    #     new_mappings = {}
    #     for g in extension.db_graphs:
    #         new_mappings[g] = self.get_mappings(g)
    #     self.mappings = new_mappings
    #     self.db_graphs = extension.db_graphs
    #
    #     # update mappings
    #     for g in self.db_graphs:
    #         tmp_mappings = []
    #         for _map in self.mappings[g]:
    #             # node of the pattern from which the extension is applied
    #             node_p = extension.pattern_node_id
    #             # mapping of the node_p in the DB graph
    #             node_db = _map.get_node_mapping(node_p)
    #             # all nodes in the DB graph that are already mapped
    #             mapped_db_nodes = set(_map.nodes_mapping().values())
    #             if extension.is_outgoing:
    #                 for neigh in set(g.successors(node_db)).difference(mapped_db_nodes):
    #                     # if the neighbor has the same labels as the extension
    #                     # and the set of edge labels of the extension is a subset of the edge labels of the neighbor
    #                     if g.get_node_labels(neigh) == extension.target_node_labels and \
    #                             all(label in g.get_edge_labels(node_db, neigh) for label in extension.target_edge_labels): # !!!!!!!!!!!!!!!FIX THIS!!!!!!!!!!!!!!!!
    #                         new_map = _map.copy()
    #                         # update node mapping
    #                         new_map.add_node_mapping(new_node_id, neigh)
    #                         # update edge mapping
    #                         key_p = 0
    #                         for label_p in self.get_edge_labels(node_p, new_node_id):
    #                             for key_db in g.edges_keys((node_db, neigh)):
    #                                 edge_db = (node_db, neigh, key_db)
    #                                 label_db = g.get_edge_label(edge_db)
    #                                 if label_p == label_db and not new_map.is_edge_mapped(edge_db):
    #                                     new_map.add_edge_mapping((node_p, new_node_id, key_p), edge_db)
    #                                     key_p += 1
    #                         tmp_mappings.append(new_map)
    #             else:
    #                 for neigh in set(g.predecessors(node_db)).difference(mapped_db_nodes):
    #                     if g.get_node_labels(neigh) == extension.target_node_labels and \
    #                             all(label in g.get_edge_labels(neigh, node_db) for label in extension.target_edge_labels):
    #                         new_map = _map.copy()
    #                         new_map.add_node_mapping(new_node_id, neigh)
    #                         key_p = 0
    #                         for label_p in self.get_edge_labels(new_node_id, node_p):
    #                             for key_db in g.edges_keys((neigh, node_db)):
    #                                 edge_db = (neigh, node_db, key_db)
    #                                 label_db = g.get_edge_label(edge_db)
    #                                 if label_p == label_db and not new_map.is_edge_mapped(edge_db):
    #                                     new_map.add_edge_mapping((new_node_id, node_p, key_p), edge_db)
    #                                     key_p += 1
    #                         tmp_mappings.append(new_map)
    #         self.mappings[g] = tmp_mappings
    #     end_time = time.time()
    #     debug_message(f"UPDATE MAPPINGS: {end_time - start_time} seconds")

    def get_mappings(self, graph):
        if self.mappings is None:
            self.mappings = {}
        if graph not in self.mappings:
            self.mappings[graph] = graph.localize(self)
        return self.mappings[graph]

    def _find_extensions(self, extension_manager: ExtensionManager) -> list[Extension]:
        """
        Generate all possible extension that if applied to the pattern, it still remains frequent.

        :param extension_manager: Object that manage the extensions
        :return: List of all possible extensions
        """
        start_time = time.time()
        debug_message("    - FIND EXTENSIONS")
        # for all graph in the database that contains the current extension
        for g in self.db_graphs:
            # obtain where the current extension is located in the graph
            mappings = self.get_mappings(g)
            # For each map we know one place where the extension is located in the graph.
            # We search all nodes that are neighbors of the current pattern and create a new extension.
            for _map in mappings:
                # get where the nodes of the extension are located in the DB graph
                f = _map.nodes_mapping()
                # retrieve nodes mapped in the DB graph
                mapped_target_nodes = set(f.values())
                # node_p  := node pattern
                # node_db := node in the DB graph mapped to node_p
                for node_p, node_db in f.items():
                    # for each node of the pattern search a possible extension
                    for neigh in set(g.successors(node_db)).difference(mapped_target_nodes):
                        extension_manager.add_extension(node_p, node_db, neigh, True, g)
                    for neigh in set(g.predecessors(node_db)).difference(mapped_target_nodes):
                        extension_manager.add_extension(node_p, neigh, node_db, False, g)

        end_time = time.time()
        debug_message(f"    - FIND EXTENSIONS: {end_time - start_time} seconds")
        return extension_manager.get_extensions()

    def __str__(self):
        global count_pattern
        output = ""
        output += f"t # {count_pattern}\n"
        for node in self.nodes(data=True):
            output += f"v {node[0]} {' '.join(node[1]['labels'])}\n"
        for edge in self.edges(data=True):
            output += f"e {edge[0]} {edge[1]} {edge[2]['type']}\n"
        output += "x " + " ".join(
            ["(" + str(g.get_name()) + ", " + str(len(self.mappings[g])) + ")" for g in self.db_graphs]) + "\n"
        output += "f " + " ".join([str(len(self.mappings[g])) for g in self.db_graphs if g in self.mappings])
        output += f"\nSupport: {len(self.db_graphs)}\n"
        output += "-------------\n"
        count_pattern += 1
        return output


def remove_redundant_patterns(patterns):
    """
    Removes patterns that are duplicates based on their code.
    """
    already_seen = set()
    new_patterns = []
    for pattern in patterns:
        code = pattern.code()
        if code not in already_seen:
            already_seen.add(code)
            new_patterns.append(pattern)
    return new_patterns


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
        self.db = []

    def mine(self):
        self._read_graphs_from_file()
        self._parse_support()
        self._adapt_target_graphs()
        if self._max_num_nodes > 2:
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
                for pattern in patterns:
                    print(pattern)
                    pattern_count += 1
            # increment the number of nodes
            curr_nodes += 1

    def _mine_patterns(self, num_nodes) -> list[Pattern]:
        debug_message("Mine patterns with", num_nodes, "nodes")
        if num_nodes == 1:
            patterns = self._mine_1node_patterns()
        elif num_nodes == 2:
            patterns = self._mine_2node_patterns()
        else:
            patterns = self._mine_next_patterns()
        self.last_mined_patterns = patterns
        return patterns

    def _mine_next_patterns(self) -> list[Pattern]:
        """
        Extends existing patterns by adding new nodes and edges based on the support threshold.
        """
        debug_message("EXTENDING", len(self.last_mined_patterns), "DIFFERENT MAPPINGS")
        patterns = []

        for i, pattern in enumerate(self.last_mined_patterns):
            debug_message("- EXTENDING PATTERN", i + 1)
            patterns.extend(pattern.extend(self.support))

        patterns = remove_redundant_patterns(patterns)

        return patterns

    def _mine_1node_patterns(self) -> list[Pattern]:
        counter = {}
        for g in self.db:
            for node in g.nodes():
                sorted_labels = g.get_node_labels(node)
                sorted_labels_str = " ".join(sorted_labels)
                if sorted_labels_str in counter:
                    counter[sorted_labels_str].add(g)
                else:
                    counter[sorted_labels_str] = {g}

        patterns = []
        for sorted_labels_str, graphs_indices in counter.items():
            if len(graphs_indices) >= self.support:
                g = MultiDiGraph()
                g.add_node(1, labels=sorted_labels_str.split(" "))
                patterns.append(Pattern(g, graphs_indices))
        return patterns

    def _mine_2node_patterns(self) -> list[Pattern]:
        # !!!!!!!!!!!!!!FIX GENERATION OF PATTERNS WITH MORE THAT 1 EDGE!!!!!!!!!!!!!!!!!!
        counter = {}
        for g in self.db:
            for src, dest in g.edges():
                src_labels = g.get_node_labels(src)
                dest_labels = g.get_node_labels(dest)
                edge_labels = g.get_edge_labels(src, dest)
                sorted_labels_str = " ".join(src_labels) + "/" + " ".join(dest_labels) + "/" + " ".join(edge_labels)
                if sorted_labels_str in counter:
                    counter[sorted_labels_str].add(g)
                else:
                    counter[sorted_labels_str] = {g}
        patterns = []
        for sorted_labels_str, graphs_indices in counter.items():
            if len(graphs_indices) >= self.support:
                src_labels, dest_labels, edge_label = sorted_labels_str.split("/")
                g = MultiDiGraph()
                g.add_node(1, labels=src_labels.split(" "))
                g.add_node(2, labels=dest_labels.split(" "))
                for edge_label in edge_label.split(" "):
                    g.add_edge(1, 2, type=edge_label)
                patterns.append(Pattern(g, graphs_indices))
        return patterns

    def _adapt_target_graphs(self):
        """
        Adapt all graphs in the database so that they all have the same labels.
        """
        # get all labels in the db
        all_node_labels = set()
        all_edge_labels = set()
        for g in self.db:
            all_node_labels = all_node_labels.union(g.get_all_node_labels())
            all_edge_labels = all_edge_labels.union(g.get_all_edge_labels())

        # for each graph in the db add the missing labels
        for g in self.db:
            missing_node_labels = all_node_labels.difference(g.get_all_node_labels())
            missing_edge_labels = all_edge_labels.difference(g.get_all_edge_labels())
            g.add_node('dummy', labels=missing_node_labels)
            for label in missing_edge_labels:
                g.add_edge('dummy', 'dummy', type=label)

    def _read_graphs_from_file(self):
        type_file = "data"
        configurator = NetworkConfigurator(type_file)
        for name, network in NetworksLoading(type_file, configurator.config).Networks.items():
            self.db.append(DBGraph(network, name))

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
            db_len = len(self.db)
            self.support = int(self.support * db_len)

    def _init_matcher(self):
        for g in self.db:
            g.init_matcher()
