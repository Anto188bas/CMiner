import os
from src.NetworkX.NetworkConfigurator import NetworkConfigurator
from src.NetworkX.NetworksLoading import NetworksLoading
from src.CMiner.MultiGraphMatch import MultiGraphMatch
from src.Graph.Graph import MultiDiGraph
from src.CMiner.BitMatrix import TargetBitMatrixOptimized, BitMatrixStrategy2

matching_total_time = 0
count_pattern = 0


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


class DBGraph(MultiDiGraph):

    def __init__(self, graph, name):
        super().__init__(graph)
        self.name = name
        self.matcher = None

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

    def add_extension(self, pattern_node_id, target_src_node_id, target_dst_node_id, is_outgoing, db_graph, _map):
        target_edge_labels_code = " ".join(db_graph.get_edge_labels(target_src_node_id, target_dst_node_id))
        if is_outgoing:
            self._add_extension_helper(self.out_extensions, pattern_node_id, target_dst_node_id, target_edge_labels_code, db_graph, _map)
        else:
            self._add_extension_helper(self.in_extensions, pattern_node_id, target_dst_node_id, target_edge_labels_code, db_graph, _map)

    def _add_extension_helper(self, extensions, pattern_node_id, target_node_id, target_edge_labels_code, db_graph, _map):
        target_node_labels_code = " ".join(db_graph.get_node_labels(target_node_id))
        extension_code = (pattern_node_id, target_node_labels_code, target_edge_labels_code)
        if extension_code not in extensions:
            extensions[extension_code] = {
                "db_graphs": set(),
                "target_node_ids_grouped_by_graphs_and_map": {}
            }
        extensions[extension_code]["db_graphs"].add(db_graph)
        if db_graph not in extensions[extension_code]["target_node_ids_grouped_by_graphs_and_map"]:
            extensions[extension_code]["target_node_ids_grouped_by_graphs_and_map"][db_graph] = {}
        if _map not in extensions[extension_code]["target_node_ids_grouped_by_graphs_and_map"][db_graph]:
            extensions[extension_code]["target_node_ids_grouped_by_graphs_and_map"][db_graph][_map] = set()
        extensions[extension_code]["target_node_ids_grouped_by_graphs_and_map"][db_graph][_map].add(target_node_id)

    def get_extensions(self) -> list['Extension']:
        out_extensions = self._filter_extensions(self.out_extensions)
        in_extensions = self._filter_extensions(self.in_extensions)
        return out_extensions + in_extensions

    def _filter_extensions(self, extensions) -> list['Extension']:
        filtered_extensions = []
        for extension_code, data in extensions.items():
            if len(data["db_graphs"]) >= self.support:
                pattern_node_id, target_node_labels_code, target_edge_labels_code = extension_code
                filtered_extensions.append(Extension(pattern_node_id, target_node_labels_code.split(" "), target_edge_labels_code.split(" "), True,
                                                     data["db_graphs"], data["target_node_ids_grouped_by_graphs_and_map"]))
        return filtered_extensions


class Extension:
    def __init__(self, pattern_node_id, target_node_labels, target_edge_labels, is_outgoing, db_graphs,
                 target_node_ids_grouped_by_graphs_and_map):
        self.pattern_node_id = pattern_node_id
        self.target_node_labels = target_node_labels
        self.target_edge_labels = target_edge_labels
        self.is_outgoing = is_outgoing
        self.db_graphs = db_graphs
        self.target_node_ids_grouped_by_graphs_and_map = target_node_ids_grouped_by_graphs_and_map


class Pattern(MultiDiGraph):

    def __init__(self, graph, db_graphs, mappings={}):
        super().__init__(graph)
        self.db_graphs = db_graphs
        self.mappings = mappings

    def __copy__(self):
        return Pattern(self, self.db_graphs, self.mappings)

    def extend(self, extension_manager) -> list['Pattern']:
        tmp_patterns = []
        already_present = set()
        for ext in self._find_extensions(extension_manager):
            new_pattern = self._apply_extension(ext)
            pattern_code = new_pattern.code()
            if pattern_code not in already_present:
                already_present.add(pattern_code)
                tmp_patterns.append(new_pattern)

        return tmp_patterns

    def get_mappings(self, graph):
        if graph not in self.mappings:
            self.mappings[graph] = graph.localize(self)
        return self.mappings[graph]

    def _apply_extension(self, extension) -> 'Pattern':
        new_pattern = self.__copy__()
        new_node_id = max(self.nodes()) + 1
        new_pattern.add_node(new_node_id, labels=extension.target_node_labels)
        for edge_label in extension.target_edge_labels:
            if extension.is_outgoing:
                new_pattern.add_edge(new_node_id, extension.pattern_node_id, type=edge_label)
            else:
                new_pattern.add_edge(extension.pattern_node_id, new_node_id, type=edge_label)

        # new_pattern.mappings = {}
        new_pattern._update_mappings(extension, new_node_id)

        return new_pattern

    def _update_mappings(self, extension, new_node_id):
        # consider only graphs that contains valid extensions
        self.db_graphs = extension.db_graphs
        # delete mappings that are not valid anymore
        # (i.e. the target graph is not in the db anymore)

        # for mapped_g in self.mappings.keys():
        #     if mapped_g not in extension.db_graphs: !!!!!!!!!!!!!!
        #         del self.mappings[mapped_g]

        # update mappings
        for g in extension.db_graphs:
            mappings = self.get_mappings(g)
            for _map in mappings:
                new_map = _map
                if _map not in extension.target_node_ids_grouped_by_graphs_and_map[g]:
                    continue
                target_node_ids = extension.target_node_ids_grouped_by_graphs_and_map[g][_map]
                for key, target_node_id in enumerate(target_node_ids):
                    if extension.is_outgoing:
                        new_map.add_node_mapping(new_node_id, target_node_id)
                        # new_map.add_edge_mapping((new_node_id, extension.pattern_node_id, key), (new_map[new_node_id], target_node_id, key))
                    else:
                        new_map.add_node_mapping(target_node_id, new_node_id)
                        # new_map.add_edge_mapping((extension.pattern_node_id, new_node_id), (target_node_id, new_map[new_node_id]))
    def _find_extensions(self, extension_manager: ExtensionManager) -> list[Extension]:
        for g in self.db_graphs:
            mappings = self.get_mappings(g)
            for _map in mappings:
                f = _map.nodes_mapping()
                mapped_target_nodes = set(f.values())
                for node_q, node_t in f.items():
                    for neigh in set(g.successors(node_t)).difference(mapped_target_nodes):
                        extension_manager.add_extension(node_q, node_t, neigh, True, g, _map)
                    for neigh in set(g.predecessors(node_t)).difference(mapped_target_nodes):
                        extension_manager.add_extension(node_q, neigh, node_t, False, g, _map)

        return extension_manager.get_extensions()

    def __str__(self):
        global count_pattern
        output = ""
        output += f"t # {count_pattern}\n"
        for node in self.nodes(data=True):
            output += f"v {node[0]} {' '.join(node[1]['labels'])}\n"
        for edge in self.edges(data=True):
            output += f"e {edge[0]} {edge[1]} {edge[2]['type']}\n"
        output += "x " + " ".join([str(g.get_name()) for g in self.db_graphs]) + "\n"
        output += f"\nSupport: {len(self.db_graphs)}\n"
        output += "-------------\n"
        count_pattern += 1
        return output


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
        self.extension_manager = ExtensionManager(support)
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
        patterns = []

        for pattern in self.last_mined_patterns:
            patterns.extend(pattern.extend(self.extension_manager))

        self._remove_redundant_patterns(patterns)

        return patterns

    def _remove_redundant_patterns(self, patterns):
        """
        Removes patterns that are duplicates based on their code.
        """
        codes = set()
        for p in patterns:
            code = p.code()
            if code not in codes:
                codes.add(code)
            else:
                patterns.remove(p)

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
