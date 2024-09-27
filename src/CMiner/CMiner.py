import copy
from NetworkX.NetworkConfigurator import NetworkConfigurator
from NetworkX.NetworksLoading import NetworksLoading
from CMiner.MultiGraphMatch import MultiGraphMatch
from Graph.Graph import MultiDiGraph
from CMiner.BitMatrix import TargetBitMatrixOptimized, BitMatrixStrategy2
from CMiner.MultiGraphMatch import MultiGraphMatch, Mapping
import time
import shutil

# TO-DO: handle multiple edges map in apply_node_extensions
# TO-DO: handle multiple edges map in apply_edge_extensions

count_pattern = 0

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


class DBGraph(MultiDiGraph):

    def __init__(self, graph, name):
        """
        Represents a graph in the database.
        """
        super().__init__(graph)
        self.name = name
        self.matcher = None

    def _init_matcher(self):
        """
        Initialize the matching algorithm.
        """
        bit_matrix = TargetBitMatrixOptimized(self, BitMatrixStrategy2())
        bit_matrix.compute()
        self.matcher = MultiGraphMatch(self, target_bit_matrix=bit_matrix)

    def localize(self, pattern) -> list['Mapping']:
        """
        Find all the mappings of the pattern in the graph.
        """
        if self.matcher is None:
            self._init_matcher()
        return self.matcher.match(pattern)

    def get_name(self):
        """
        Return the name of the graph
        """
        return self.name


class PatternMappings:

    def __init__(self):
        """
        Keep track of each mapping for each graph of a specific pattern.
        """
        self.patterns_mappings = {}

    def graphs(self) -> list[DBGraph]:
        """
        Return the graphs that contains the pattern
        """
        return list(self.patterns_mappings.keys())

    def mappings(self, graph) -> list[Mapping]:
        """
        Return the mappings of the pattern in the graph.
        """
        return self.patterns_mappings[graph]

    def set_mapping(self, graph, mappings: [Mapping]):
        """
        Set the mappings of the pattern in the graph.
        """
        self.patterns_mappings[graph] = mappings


class EdgeExtension:

    def __init__(self, pattern_node_src, pattern_node_dest, edge_labels, db_graphs, extension_edge_ids):
        self.pattern_node_src = pattern_node_src
        self.pattern_node_dest = pattern_node_dest
        self.edge_labels = edge_labels
        self.db_graphs = db_graphs
        self.extension_edge_ids = extension_edge_ids

    def __copy__(self):
        return EdgeExtension(
            self.pattern_node_src,
            self.pattern_node_dest,
            self.edge_labels,
            self.db_graphs,
            self.extension_edge_ids
        )


class EdgeExtensionManager:

    def __init__(self, support):
        self.min_support = support
        self.extensions = {}
        self.extensions_node_ids = {}

    def add(self, pattern_node_src, pattern_node_dest, db_graph, _map):
        """
        Add an extension to the manager.
        """
        target_src_node_id = _map.node(pattern_node_src)
        target_dst_node_id = _map.node(pattern_node_dest)
        target_edge_labels_code = " ".join(db_graph.get_edge_labels(target_src_node_id, target_dst_node_id))

        extension_code = (pattern_node_src, pattern_node_dest, target_edge_labels_code)
        if extension_code not in self.extensions:
            self.extensions[extension_code] = set()
        self.extensions[extension_code].add(db_graph)

        if extension_code not in self.extensions_node_ids:
            self.extensions_node_ids[extension_code] = {}
        if db_graph not in self.extensions_node_ids[extension_code]:
            self.extensions_node_ids[extension_code][db_graph] = {}
        if _map not in self.extensions_node_ids[extension_code][db_graph]:
            self.extensions_node_ids[extension_code][db_graph][_map] = set()
        self.extensions_node_ids[extension_code][db_graph][_map].add((target_src_node_id, target_dst_node_id))

    def frequent_extensions(self) -> list['EdgeExtension']:
        """
        Return a list of Extensions that are frequent
        """
        filtered_extensions = []
        for (pattern_node_src, pattern_node_dest, target_edge_labels), db_graphs in self.extensions.items():
            if len(db_graphs) >= self.min_support:
                extensions_edge_ids = self.extensions_node_ids[
                    (pattern_node_src, pattern_node_dest, target_edge_labels)]
                target_edge_labels = target_edge_labels.split(" ")
                ext = EdgeExtension(pattern_node_src, pattern_node_dest, target_edge_labels, db_graphs,
                                    extensions_edge_ids)
                filtered_extensions.append(ext)
        return filtered_extensions


class NodeExtension:

    def __init__(self, pattern_node_id, target_node_labels, target_edge_labels, is_outgoing, db_graphs,
                 extension_node_ids):
        self.pattern_node_id = pattern_node_id
        self.target_node_labels = target_node_labels
        self.target_edge_labels = target_edge_labels
        self.is_outgoing = is_outgoing
        self.db_graphs = db_graphs
        self.extension_node_ids = extension_node_ids


class NodeExtensionManager:

    def __init__(self, support):
        self.min_support = support
        self.out_extensions = {}
        self.in_extensions = {}
        self.out_extensions_node_ids = {}
        self.in_extensions_node_ids = {}

    def add(self, pattern_node_id, target_src_node_id, target_dst_node_id, is_outgoing, db_graph, _map):
        """
        Add an extension to the manager
        """
        target_edge_labels_code = " ".join(db_graph.get_edge_labels(target_src_node_id, target_dst_node_id))
        if is_outgoing:
            self._add_helper(self.out_extensions, self.out_extensions_node_ids, pattern_node_id,
                                       target_dst_node_id, target_edge_labels_code, db_graph, _map)
        else:
            self._add_helper(self.in_extensions, self.in_extensions_node_ids, pattern_node_id,
                                       target_src_node_id, target_edge_labels_code, db_graph, _map)

    @staticmethod
    def _add_helper(extensions, extensions_node_ids, pattern_node_id, target_node_id,
                              target_edge_labels_code, db_graph, _map):
        """
        Helper function to add an extension to the manager. It is used to avoid code duplication.
        """
        target_node_labels_code = " ".join(db_graph.get_node_labels(target_node_id))
        extension_code = (pattern_node_id, target_node_labels_code, target_edge_labels_code)
        if extension_code not in extensions:
            extensions[extension_code] = set()
        extensions[extension_code].add(db_graph)
        if extension_code not in extensions_node_ids:
            extensions_node_ids[extension_code] = {}
        if db_graph not in extensions_node_ids[extension_code]:
            extensions_node_ids[extension_code][db_graph] = {}
        if _map not in extensions_node_ids[extension_code][db_graph]:
            extensions_node_ids[extension_code][db_graph][_map] = set()
        extensions_node_ids[extension_code][db_graph][_map].add(target_node_id)

    def frequent_extensions(self) -> list['NodeExtension']:
        """
        Return a list of Extensions that are frequent
        """
        out_extensions = self._frequent_extensions_helper(self.out_extensions, self.out_extensions_node_ids, True)
        in_extensions = self._frequent_extensions_helper(self.in_extensions, self.in_extensions_node_ids, False)
        return out_extensions + in_extensions

    def _frequent_extensions_helper(self, extensions, extensions_node_ids, is_outgoing) -> list['NodeExtension']:
        """
        Helper function to return a list of Extensions that are frequent. It is used to avoid code duplication.
        """
        filtered_extensions = []
        for (pattern_node_id, target_node_labels, target_edge_labels), db_graphs in extensions.items():

            if len(db_graphs) >= self.min_support:
                extensions_node_id = extensions_node_ids[(pattern_node_id, target_node_labels, target_edge_labels)]
                target_node_labels = target_node_labels.split(" ")
                target_edge_labels = target_edge_labels.split(" ")
                ext = NodeExtension(pattern_node_id, target_node_labels, target_edge_labels, is_outgoing, db_graphs,
                                    extensions_node_id)
                filtered_extensions.append(ext)
        return filtered_extensions


class Pattern(MultiDiGraph):

    def __init__(self, pattern_mappings, extension_applied: NodeExtension = None, extended_pattern: 'Pattern' = None,
                 **attr):
        """
        Represents a pattern in the database.
        """
        if extended_pattern is not None:
            super().__init__(extended_pattern, **attr)
        else:
            super().__init__(**attr)
        self.extended_pattern = extended_pattern
        self.pattern_mappings = pattern_mappings
        self.extension_applied = extension_applied

    def graphs(self) -> list[DBGraph]:
        """
        Return the graphs that contains the pattern
        """
        return self.pattern_mappings.graphs()

    def support(self):
        """
        Return the support of the pattern
        """
        return len(self.graphs())

    def find_node_extensions(self, extension_manager: NodeExtensionManager) -> list[NodeExtension]:
        """
        Generate all possible node extension that if applied to the pattern, it still remains frequent.
        """
        # for all graph in the database that contains the current extension
        for g in self.graphs():
            # obtain where the current extension is located in the graph
            mappings = self.pattern_mappings.mappings(g)
            # For each map we know one place where the extension is located in the graph.
            # We search all nodes that are neighbors of the current pattern and create a new extension.
            for _map in mappings:
                # retrieve nodes mapped in the DB graph
                mapped_target_nodes = _map.nodes()
                # node_p  := node pattern
                # node_db := node in the DB graph mapped to node_p
                for node_p, node_db in _map.node_pairs():
                    # for each node of the pattern search a possible extension
                    for neigh in set(g.successors(node_db)).difference(mapped_target_nodes):
                        extension_manager.add(node_p, node_db, neigh, True, g, _map)
                    for neigh in set(g.predecessors(node_db)).difference(mapped_target_nodes):
                        extension_manager.add(node_p, neigh, node_db, False, g, _map)

        return extension_manager.frequent_extensions()

    def find_edge_extensions(self, extension_manager: EdgeExtensionManager) -> list[list['EdgeExtension']]:
        """
        Generate all possible edge extension that if applied to the pattern, it still remains frequent.
        """
        for g in self.graphs():
            for _map in self.pattern_mappings.mappings(g):
                # subgraph of the projected pattern (include also edges not mapped with the patten)
                mapped_pattern_complete_graph = g.subgraph(_map.nodes())
                mapped_pattern_complete_graph_edges = set(mapped_pattern_complete_graph.edges(keys=True))
                mapped_pattern_edges = set(_map.get_target_edges())
                candidate_edges = mapped_pattern_complete_graph_edges.difference(mapped_pattern_edges)
                inverse_map = _map.inverse()
                for edge in candidate_edges:
                    pattern_node_src = inverse_map.node(edge[0])
                    pattern_node_dest = inverse_map.node(edge[1])
                    extension_manager.add(pattern_node_src, pattern_node_dest, g, _map)

        extensions = extension_manager.frequent_extensions()

        graphs = sorted(self.graphs(), key=lambda x: x.get_name())
        extension_matrix = [[0 for _ in range(len(graphs))] for _ in range(len(extensions))]
        for i, ext in enumerate(extensions):
            for j, g in enumerate(graphs):
                if g in ext.db_graphs:
                    extension_matrix[i][j] = 1

        # group row by row
        matrix_indices_grouped = {}
        for i, row in enumerate(extension_matrix):
            row_code = "".join(map(str, row))
            if row_code not in matrix_indices_grouped:
                matrix_indices_grouped[row_code] = []
            matrix_indices_grouped[row_code].append(i)


        groups = []
        for row_code, indices in matrix_indices_grouped.items():
            columns_to_select = [i for i, v in enumerate(row_code) if v == "1"]
            group = []
            for i, ext in enumerate(extensions):
                if all(extension_matrix[i][j] == 1 for j in columns_to_select):
                    ext_copy = ext.__copy__()
                    ext_copy.db_graphs = [graphs[j] for j in columns_to_select]
                    group.append(ext_copy)
            groups.append(group)

        return groups

    def apply_node_extension(self, extension) -> 'Pattern':
        """
        Apply the node extension to the pattern.
        """
        # Object to keep track of the new pattern mappings
        new_pattern_mappings = PatternMappings()
        # The id of the previous pattern node that is extended
        pattern_node_id = extension.pattern_node_id

        # Apply extension to the pattern (add node and edges)
        new_pattern = Pattern(extended_pattern=self, pattern_mappings=new_pattern_mappings, extension_applied=extension)
        new_pattern_new_node_id = int(len(new_pattern.nodes())) + 1
        new_pattern.add_node(new_pattern_new_node_id, labels=extension.target_node_labels)
        for target_edge_label in extension.target_edge_labels:
            if extension.is_outgoing:
                new_pattern.add_edge(pattern_node_id, new_pattern_new_node_id, type=target_edge_label)
            else:
                new_pattern.add_edge(new_pattern_new_node_id, pattern_node_id, type=target_edge_label)

        # Update the pattern mappings
        for target in extension.db_graphs:
            new_mappings = []
            for target_map in self.pattern_mappings.mappings(target):
                # Check if the extension is applicable by verifying if target_map is in extension_node_ids[target]
                # (it means that the extension is applicable because it let us discover a new extension)
                if target_map not in extension.extension_node_ids[target]:
                    continue

                target_node_ids = extension.extension_node_ids[target][target_map]

                for target_node_id in target_node_ids:
                    # node mapping
                    node_mapping = {new_pattern_new_node_id: target_node_id}
                    # edge mapping
                    if extension.is_outgoing:
                        new_pattern_edge = (pattern_node_id, new_pattern_new_node_id, 0)
                        target_edge_src = target_map.nodes_mapping()[pattern_node_id]
                        target_edge_dest = target_node_id
                        target_edge = (target_edge_src, target_edge_dest,
                                       target.edge_keys(target_edge_src, target_edge_dest)[0])  # change
                    else:
                        new_pattern_edge = (new_pattern_new_node_id, pattern_node_id, 0)
                        target_edge_src = target_node_id
                        target_edge_dest = target_map.nodes_mapping()[pattern_node_id]
                        target_edge = (target_edge_src, target_edge_dest,
                                       target.edge_keys(target_edge_src, target_edge_dest)[0])  # change
                    edge_mapping = {new_pattern_edge: target_edge}
                    new_mapping = Mapping(node_mapping=node_mapping, edge_mapping=edge_mapping,
                                          extended_mapping=target_map)
                    new_mappings.append(new_mapping)
            new_pattern_mappings.set_mapping(target, new_mappings)
        return new_pattern



    def apply_edge_extension(self, extensions: list[EdgeExtension]) -> 'Pattern':
        """
        Apply the edge extension to the pattern.
        """

        # Object to keep track of the new pattern mappings
        db_graphs = extensions[0].db_graphs  # NOTE: all extensions are in the same graphs
        new_pattern_mappings = PatternMappings()

        # Apply extension to the pattern (add edges)
        new_pattern = Pattern(extended_pattern=self, pattern_mappings=new_pattern_mappings)

        for ext in extensions:
            new_pattern.add_edge(ext.pattern_node_src, ext.pattern_node_dest,
                                 type=ext.edge_labels[0])  # FIX ext.edge_labels[0]

        # Update the pattern mappings
        for target in db_graphs:
            new_mappings = []
            for target_map in self.pattern_mappings.mappings(target):
                if any(target_map not in ext.extension_edge_ids[target] for ext in extensions):
                    continue
                new_mapping = Mapping(extended_mapping=target_map)
                for ext in extensions:
                    target_edge_ids = ext.extension_edge_ids[target][target_map]
                    for target_edge_src, target_edge_dest in target_edge_ids:
                        new_pattern_edge = (ext.pattern_node_src, ext.pattern_node_dest, 0)  # TO-DO: SET A KEY
                        target_edge = (target_edge_src, target_edge_dest,
                                       target.edge_keys(target_edge_src, target_edge_dest)[0])  # FIX
                        new_mapping.set_edge(new_pattern_edge, target_edge)
                new_mappings.append(new_mapping)
            new_pattern_mappings.set_mapping(target, new_mappings)
        return new_pattern

    def __str__(self, show_mappings=False):
        global count_pattern
        output = ""
        # graph info
        output += f"t # {count_pattern}\n"
        for node in self.nodes(data=True):
            output += f"v {node[0]} {' '.join(node[1]['labels'])}\n"
        for edge in self.edges(data=True):
            output += f"e {edge[0]} {edge[1]} {edge[2]['type']}\n"
        # support
        output += f"s {self.support()}\n"
        output += f"f {sum(len(self.pattern_mappings.mappings(g)) for g in self.graphs())}\n"

        if show_mappings:
            for g in self.graphs():
                # graph_name frequency
                output += g.get_name() + " " + str(len(self.pattern_mappings.mappings(g))) + " "
                # ({nodes map}, {edges map}), ...
                output += " ".join(
                    [str(_map) for _map in self.pattern_mappings.mappings(g)])
                output += "\n"
        else:
            #frequency info
            frequencies = ["(" + g.get_name() + ", " + str(len(self.pattern_mappings.mappings(g))) + ")" for g in self.graphs()]
            output += "x " + " ".join(frequencies) + "\n"
        output += "-" + "\n"

        count_pattern += 1
        return output


class CMiner:

    def __init__(self,
                 db_file,
                 support,
                 min_nodes=1,
                 max_nodes=float('inf'),
                 start_patterns=None,  # implement
                 show_mappings=False,
                 output_path=None
                 ):
        self.db_file = db_file
        self._min_nodes = min_nodes
        self._max_nodes = max_nodes
        self.min_support = support
        self._start_patterns = start_patterns
        self.db = []
        self.show_mappings = show_mappings
        self.output_path = output_path

    def find_start_patterns(self) -> [Pattern]:

        if self._start_patterns is None:
            return self._mine_1node_patterns()

        start_patterns = []
        found_mappings = {}

        for p in self._start_patterns:
            print(p)
            for g in self.db:
                matching = g.localize(p)
                if len(matching) > 0:
                    found_mappings[g] = matching

        for p in self._start_patterns:
            pattern_mappings = PatternMappings()
            for g, mappings in found_mappings.items():
                pattern_mappings.set_mapping(g, mappings)
            start_patterns.append(Pattern(extended_pattern=p, pattern_mappings=pattern_mappings))

        return start_patterns

    def mine(self):
        self._read_graphs_from_file()
        self._parse_support()

        pattern_codes = set()

        # Stack for DFS
        stack = self.find_start_patterns()


        # Open the file initially
        output_file = None
        if self.output_path is not None:
            output_file = open(self.output_path, "a")

        while len(stack) > 0:
            pattern_to_extend = stack.pop()

            # print pattern to console and file if it meets the min node requirement
            if len(pattern_to_extend.nodes()) >= self._min_nodes:
                if self.output_path is not None:
                    print("New solution")
                    print(pattern_to_extend.__str__(self.show_mappings), file=output_file)
                else:
                    print(pattern_to_extend.__str__(self.show_mappings))

            # Check if the pattern is already at the max number of nodes
            if len(pattern_to_extend.nodes()) >= self._max_nodes:
                del pattern_to_extend
                continue

            # Find extensions
            node_extensions = pattern_to_extend.find_node_extensions(NodeExtensionManager(self.min_support))

            if len(node_extensions) == 0:
                del pattern_to_extend
                # Backtracking occurs when no more extensions are found
                if self.output_path is not None:
                    output_file.close()  # Close the file on backtrack
                    output_file = open(self.output_path, "a")  # Reopen the file
                continue

            for node_ext in node_extensions:
                new_tree_pattern = pattern_to_extend.apply_node_extension(node_ext)
                new_pattern_code = new_tree_pattern.code()

                # Ensure no duplicate patterns are processed
                if new_pattern_code not in pattern_codes:

                    tree_pattern_added = False

                    edge_extensions = new_tree_pattern.find_edge_extensions(EdgeExtensionManager(self.min_support))

                    # If no edge extensions are found, add the tree pattern to the stack
                    if len(edge_extensions) == 0:
                        stack.append(new_tree_pattern)
                        pattern_codes.add(new_pattern_code)
                        continue

                    graphs_covered_by_edge_extensions = set()
                    for edge_ext in edge_extensions:
                        for g in edge_ext[0].db_graphs:
                            graphs_covered_by_edge_extensions.add(g)

                    for edge_ext in edge_extensions:
                        new_cycle_pattern = new_tree_pattern.apply_edge_extension(edge_ext)


                        # If the support of the tree pattern is greater than the cycle pattern
                        # it means that the tree cannot be closed in a cycle for all of his
                        # occurrence in each graph, so it's considered the tree pattern and added to the stack.
                        # Also check if the pattern is not already in the stack, because the same tree can be
                        # considered with more than one edge extension.
                        if (not tree_pattern_added) and (
                                new_tree_pattern.support() > len(graphs_covered_by_edge_extensions)) and (
                                new_tree_pattern.support() > new_cycle_pattern.support()):
                            stack.append(new_tree_pattern)
                            pattern_codes.add(new_pattern_code)
                            tree_pattern_added = True

                        new_cycle_pattern_code = new_cycle_pattern.code()
                        if new_cycle_pattern_code not in pattern_codes:
                            stack.append(new_cycle_pattern)
                            pattern_codes.add(new_cycle_pattern_code)
                        else:
                            del new_cycle_pattern
                else:
                    del new_tree_pattern

        # Close the file if it was opened
        if output_file is not None:
            output_file.close()

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

        # update the mappings
        patterns = []
        for sorted_labels_str, graphs in counter.items():
            if len(graphs) >= self.min_support:
                pattern_mappings = PatternMappings()
                p = Pattern(pattern_mappings)
                p.add_node(0, labels=sorted_labels_str.split(" "))
                for g in graphs:
                    p.pattern_mappings.set_mapping(g, [Mapping(node_mapping={0: node}) for node in g.nodes() if
                                                       g.get_node_labels(node) == sorted_labels_str.split(" ")])
                patterns.append(p)
        return patterns

    def _read_graphs_from_file(self):
        type_file = self.db_file.split('.')[-1]
        configurator = NetworkConfigurator(self.db_file, type_file)
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
        if self.min_support <= 1:
            db_len = len(self.db)
            self.min_support = int(self.min_support * db_len)

