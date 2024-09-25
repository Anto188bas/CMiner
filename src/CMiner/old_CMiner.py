import copy

from NetworkX.NetworkConfigurator import NetworkConfigurator
from NetworkX.NetworksLoading import NetworksLoading
from CMiner.MultiGraphMatch import MultiGraphMatch
from Graph.Graph import MultiDiGraph
from CMiner.BitMatrix import TargetBitMatrixOptimized, BitMatrixStrategy2
from CMiner.MultiGraphMatch import MultiGraphMatch
import time
import shutil

# TO-DO: handle multiple edges map in apply_node_extensions
# TO-DO: handle multiple edges map in apply_edge_extensions

count_pattern = 0


def debug_message(*args, **kwargs):
    if False:
        print(*args, **kwargs)


def get_terminal_width():
    # Restituisce la larghezza del terminale o 80 come default
    return shutil.get_terminal_size().columns


def generate_separator():
    width = get_terminal_width()
    return '-' * width


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


class Mapping:

    def __init__(self, node_mapping=None, edge_mapping=None, extended_mapping: 'Mapping' = None):
        if edge_mapping is None:
            edge_mapping = {}
        if node_mapping is None:
            node_mapping = {}
        self.extended_mapping = extended_mapping
        self.node_mapping = node_mapping
        self.edge_mapping = edge_mapping

    def nodes_mapping(self) -> dict:
        node_mapping = {}
        if self.extended_mapping is not None:
            node_mapping.update(self.extended_mapping.nodes_mapping())
        node_mapping.update(self.node_mapping)
        return node_mapping

    def edges_mapping(self) -> dict:
        edge_mapping = {}
        if self.extended_mapping is not None:
            edge_mapping.update(self.extended_mapping.edges_mapping())
        edge_mapping.update(self.edge_mapping)
        return edge_mapping

    # def get_target_nodes(self):
    #     return self.node_mapping.values()
    #
    def get_target_edges(self) -> list:
        if self.extended_mapping is not None:
            edges = list(self.extended_mapping.get_target_edges())
            edges.extend(self.edge_mapping.values())
            return edges
        return list(self.edge_mapping.values())

    def get_pattern_nodes(self):
        return self.node_mapping.keys()

    def get_pattern_edges(self):
        return self.edge_mapping.keys()

    def __str__(self):
        return f"({{{self.get_node_mapping_str()}}}, {{{self.get_edge_mapping_str()}}})"

    def get_node_mapping_str(self):
        node_mapping_str = ""
        if self.extended_mapping is not None:
            node_mapping_str += self.extended_mapping.get_node_mapping_str() + ", "
        node_mapping_str += " ".join([f"{k}->{v}" for k, v in self.node_mapping.items()])
        return node_mapping_str

    def get_edge_mapping_str(self):
        if len(self.edge_mapping) == 0:
            return ""
        edge_mapping_str = ""
        if self.extended_mapping is not None:
            previous_edge_mapping_str = self.extended_mapping.get_edge_mapping_str()
            edge_mapping_str += previous_edge_mapping_str + (", " if len(previous_edge_mapping_str) > 0 else "")
        edge_mapping_str += " ".join([f"{k}->{v}" for k, v in self.edge_mapping.items()])
        return edge_mapping_str

    def set_edge(self, pattern_edge, target_edge):
        self.edge_mapping[pattern_edge] = target_edge


class PatternMappings:

    def __init__(self, graphs, patterns_mappings=None):
        self.graphs = graphs
        if patterns_mappings is None:
            patterns_mappings = {}
        self.patterns_mappings = patterns_mappings

    def get_graphs(self) -> list[DBGraph]:
        return self.graphs

    def set_graphs(self, graphs):
        self.graphs = graphs

    def get_mappings(self, graph) -> list[Mapping]:
        return self.patterns_mappings[graph]

    def set_mapping(self, graph, mappings: [Mapping]):
        self.patterns_mappings[graph] = mappings

    def support(self):
        return len(self.graphs)


class EdgeExtension:

    def __init__(self, pattern_node_src, pattern_node_dest, edge_labels, db_graphs, extension_edge_ids):
        self.pattern_node_src = pattern_node_src
        self.pattern_node_dest = pattern_node_dest
        self.edge_labels = edge_labels
        self.db_graphs = db_graphs
        self.extension_edge_ids = extension_edge_ids

    def __copy__(self):
        return EdgeExtension(self.pattern_node_src, self.pattern_node_dest, self.edge_labels, self.db_graphs,
                             self.extension_edge_ids)

    def __str__(self):
        output = f"({self.pattern_node_src} -- {self.edge_labels} --> {self.pattern_node_dest}) "

        output += " ".join([g.name for g in self.db_graphs])
        # output += str(self.extension_edge_ids)
        return output


class EdgeExtensionManager:

    def __init__(self, support):
        self.min_support = support
        self.extensions = {}
        self.extensions_node_ids = {}

    def add_extension(self, pattern_node_src, pattern_node_dest, db_graph, _map):
        target_src_node_id = _map.nodes_mapping()[pattern_node_src]
        target_dst_node_id = _map.nodes_mapping()[pattern_node_dest]
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

    def get_extensions(self) -> list['EdgeExtension']:
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

    def __str__(self):
        if self.is_outgoing:
            output = f"({self.pattern_node_id} -- {self.target_edge_labels} --> {self.target_node_labels}) "
        else:
            output = f"({self.pattern_node_id} <-- {self.target_edge_labels} -- {self.target_node_labels}) "
        # output += str(self.extension_node_ids)
        return output


class NodeExtensionManager:

    def __init__(self, support):
        self.min_support = support
        self.out_extensions = {}
        self.in_extensions = {}
        self.out_extensions_node_ids = {}
        self.in_extensions_node_ids = {}

    def add_extension(self, pattern_node_id, target_src_node_id, target_dst_node_id, is_outgoing, db_graph, _map):
        target_edge_labels_code = " ".join(db_graph.get_edge_labels(target_src_node_id, target_dst_node_id))
        if is_outgoing:
            self._add_extension_helper(self.out_extensions, self.out_extensions_node_ids, pattern_node_id,
                                       target_dst_node_id, target_edge_labels_code, db_graph, _map)
        else:
            self._add_extension_helper(self.in_extensions, self.in_extensions_node_ids, pattern_node_id,
                                       target_src_node_id, target_edge_labels_code, db_graph, _map)

    def _add_extension_helper(self, extensions, extensions_node_ids, pattern_node_id, target_node_id,
                              target_edge_labels_code, db_graph, _map):
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

    def get_extensions(self) -> list['NodeExtension']:
        out_extensions = self._filter_extensions(self.out_extensions, self.out_extensions_node_ids, True)
        in_extensions = self._filter_extensions(self.in_extensions, self.in_extensions_node_ids, False)
        return out_extensions + in_extensions

    def _filter_extensions(self, extensions, extensions_node_ids, is_outgoing) -> list['NodeExtension']:
        """
        Return a list of Extensions that are frequent
        :param extensions:
        :return:
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

        if extended_pattern is not None:
            # Copia i nodi e gli archi dal pattern esteso
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
        return self.pattern_mappings.get_graphs()

    def support(self):
        """
        Return the support of the pattern
        """
        return len(self.graphs())

    def find_node_extensions(self, extension_manager: NodeExtensionManager) -> list[NodeExtension]:
        """
        Generate all possible extension that if applied to the pattern, it still remains frequent.

        :param extension_manager: Object that manage the extensions
        :return: List of all possible extensions
        """
        # for all graph in the database that contains the current extension
        for g in self.graphs():
            # obtain where the current extension is located in the graph
            mappings = self.pattern_mappings.get_mappings(g)
            # For each map we know one place where the extension is located in the graph.
            # We search all nodes that are neighbors of the current pattern and create a new extension.
            for _map in mappings:
                # get where the nodes of the extension are located in the DB graph
                node_mapping = _map.nodes_mapping()
                # retrieve nodes mapped in the DB graph
                mapped_target_nodes = set(node_mapping.values())
                # node_p  := node pattern
                # node_db := node in the DB graph mapped to node_p
                for node_p, node_db in node_mapping.items():
                    # for each node of the pattern search a possible extension
                    for neigh in set(g.successors(node_db)).difference(mapped_target_nodes):
                        extension_manager.add_extension(node_p, node_db, neigh, True, g, _map)
                    for neigh in set(g.predecessors(node_db)).difference(mapped_target_nodes):
                        extension_manager.add_extension(node_p, neigh, node_db, False, g, _map)

        return extension_manager.get_extensions()

    def find_edge_extensions(self, extension_manager: EdgeExtensionManager) -> list[list['EdgeExtension']]:

        inverse_mappings_nodes = {}
        for g in self.graphs():
            for _map in self.pattern_mappings.get_mappings(g):
                if g not in inverse_mappings_nodes:
                    inverse_mappings_nodes[g] = {}
                inverse_mappings_nodes[g][_map] = {v: k for k, v in _map.nodes_mapping().items()}

        for g in self.graphs():
            for _map in self.pattern_mappings.get_mappings(g):
                # subgraph of the projected pattern (include also edges not mapped with the patten)
                mapped_pattern_complete_graph = g.subgraph([_map.nodes_mapping()[node] for node in self.nodes()])
                mapped_pattern_complete_graph_edges = set(mapped_pattern_complete_graph.edges(keys=True))
                mapped_pattern_edges = set(_map.get_target_edges())
                candidate_edges = mapped_pattern_complete_graph_edges.difference(mapped_pattern_edges)
                for edge in candidate_edges:
                    pattern_node_src = inverse_mappings_nodes[g][_map][edge[0]]
                    pattern_node_dest = inverse_mappings_nodes[g][_map][edge[1]]
                    extension_manager.add_extension(pattern_node_src, pattern_node_dest, g, _map)

        extensions = extension_manager.get_extensions()

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

        debug_message(matrix_indices_grouped)

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
        # Object to keep track of the new pattern mappings
        new_pattern_mappings = PatternMappings(extension.db_graphs)
        # The id of the previous pattern node that is extended
        pattern_node_id = extension.pattern_node_id

        # Apply extension to the pattern (add node and edges)
        new_pattern = Pattern(extended_pattern=self, pattern_mappings=new_pattern_mappings, extension_applied=extension)
        new_pattern_new_node_id = len(new_pattern.nodes())
        new_pattern.add_node(new_pattern_new_node_id, labels=extension.target_node_labels)
        for target_edge_label in extension.target_edge_labels:
            if extension.is_outgoing:
                new_pattern.add_edge(pattern_node_id, new_pattern_new_node_id, type=target_edge_label)
            else:
                new_pattern.add_edge(new_pattern_new_node_id, pattern_node_id, type=target_edge_label)

        # Update the pattern mappings
        for target in extension.db_graphs:
            new_mappings = []
            for target_map in self.pattern_mappings.get_mappings(target):
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

        # for s, d in self.edges():
        #     for ext in extensions:
        #         src = ext.pattern_node_src
        #         dst = ext.pattern_node_dest
        #         if s == src and d == dst:
        #             return None

        # Object to keep track of the new pattern mappings
        db_graphs = extensions[0].db_graphs  # NOTE: all extensions are in the same graphs
        new_pattern_mappings = PatternMappings(db_graphs)

        # Apply extension to the pattern (add edges)
        new_pattern = Pattern(extended_pattern=self, pattern_mappings=new_pattern_mappings)

        for ext in extensions:
            new_pattern.add_edge(ext.pattern_node_src, ext.pattern_node_dest,
                                 type=ext.edge_labels[0])  # FIX ext.edge_labels[0]

        # Update the pattern mappings
        for target in db_graphs:
            new_mappings = []
            for target_map in self.pattern_mappings.get_mappings(target):
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
        output += f"s {self.pattern_mappings.support()}\n"
        # output += f"f {sum(len(self.pattern_mappings.get_mappings(g)) for g in self.graphs())}\n"

        # if show_mappings:
        #     for g in self.graphs():
        #         # graph_name frequency
        #         output += g.get_name() + " " + str(len(self.pattern_mappings.get_mappings(g))) + " "
        #         # ({nodes map}, {edges map}), ...
        #         output += " ".join(
        #             [str(_map) for _map in self.pattern_mappings.get_mappings(g)])
        #         output += "\n"
        # else:
        #     #frequency info
        #     frequencies = ["(" + g.get_name() + ", " + str(len(self.pattern_mappings.get_mappings(g))) + ")" for g in self.graphs()]
        #     output += "x " + " ".join(frequencies) + "\n"
        # output += generate_separator() + "\n"
        output += "--------------" + "\n"

        count_pattern += 1
        return output


class CMiner:

    def __init__(self,
                 db_file,
                 support,
                 min_nodes=1,
                 max_nodes=float('inf'),
                 start_pattern=None,  # implement
                 show_mappings=False,
                 output_path=None
                 ):
        self.db_file = db_file
        self._min_nodes = min_nodes
        self._max_nodes = max_nodes
        self.min_support = support
        self._start_pattern = start_pattern
        self.db = []
        self.show_mappings = show_mappings
        self.output_path = output_path

    def find_start_patterns(self) -> [Pattern]:

        if self._start_pattern is None:
            return self._mine_1node_patterns()

        start_patterns = []
        found_mappings = {}
        self._init_matcher()
        for g in self.db:
            matchings = g.localize(self._start_pattern)
            if len(matchings) > 0:
                found_mappings[g] = matchings
        pattern_mappings = PatternMappings(list(found_mappings.keys()))
        for g, mappings in found_mappings.items():
            pattern_mappings.set_mapping(g, mappings)

        new_tree_pattern = Pattern(extended_pattern=self._start_pattern, pattern_mappings=pattern_mappings)

        pattern_codes = set()
        tree_pattern_added = False

        edge_extensions = new_tree_pattern.find_edge_extensions(EdgeExtensionManager(self.min_support))

        for edge_ext in edge_extensions:
            new_cycle_pattern = new_tree_pattern.apply_edge_extension(edge_ext)
            if (not tree_pattern_added) and (new_tree_pattern.support() > new_cycle_pattern.support()):
                pattern_codes.add(new_tree_pattern.code())
                tree_pattern_added = True

            new_cycle_pattern_code = new_cycle_pattern.code()
            if new_cycle_pattern_code not in pattern_codes:
                pattern_codes.add(new_cycle_pattern_code)
                start_patterns.append(new_cycle_pattern)

        if len(edge_extensions) == 0:
            start_patterns.append(new_tree_pattern)

        return start_patterns

    def mine(self):
        self._read_graphs_from_file()
        self._parse_support()

        pattern_codes = set()

        # Stack for DFS
        stack = self.find_start_patterns()

        debug_message("Pattern iniziali:", len(stack), "\n\n")

        # Open the file initially
        output_file = None
        if self.output_path is not None:
            output_file = open(self.output_path, "a")

        while len(stack) > 0:
            pattern_to_extend = stack.pop()

            debug_message("ESTRAZIONE:")
            debug_message(pattern_to_extend)

            # print pattern to console and file if it meets the min node requirement
            if len(pattern_to_extend.nodes()) >= self._min_nodes:
                if self.output_path is not None:
                    print("New solution")
                    print(pattern_to_extend.__str__(self.show_mappings), file=output_file)
                else:
                    print(pattern_to_extend.__str__(self.show_mappings))

            # Check if the pattern is already at the max number of nodes
            if len(pattern_to_extend.nodes()) >= self._max_nodes:
                debug_message("BACKTRACKING: numero massimo di nodi")
                del pattern_to_extend
                continue

            # Find extensions
            node_extensions = pattern_to_extend.find_node_extensions(NodeExtensionManager(self.min_support))

            if len(node_extensions) == 0:
                debug_message("BACKTRACKING: non ha estensioni di nodi")
                del pattern_to_extend
                # Backtracking occurs when no more extensions are found
                if self.output_path is not None:
                    output_file.close()  # Close the file on backtrack
                    output_file = open(self.output_path, "a")  # Reopen the file
                continue

            debug_message("ESTENSIONI NODI: ", len(node_extensions))
            for node_ext in node_extensions:
                new_tree_pattern = pattern_to_extend.apply_node_extension(node_ext)
                debug_message("APPLICO L'ESTENSIONE NODI:", node_ext)
                debug_message(new_tree_pattern)
                new_pattern_code = new_tree_pattern.code()

                # Ensure no duplicate patterns are processed
                if new_pattern_code not in pattern_codes:

                    tree_pattern_added = False

                    edge_extensions = new_tree_pattern.find_edge_extensions(EdgeExtensionManager(self.min_support))

                    # If no edge extensions are found, add the tree pattern to the stack
                    if len(edge_extensions) == 0:
                        debug_message("Non ha estensioni di archi, aggiungo il pattern alla coda")
                        stack.append(new_tree_pattern)
                        pattern_codes.add(new_pattern_code)
                        continue

                    debug_message("ESTENSIONI ARCHI: ", len(edge_extensions))
                    graphs_covered_by_edge_extensions = set()
                    for edge_ext in edge_extensions:
                        for g in edge_ext[0].db_graphs:
                            graphs_covered_by_edge_extensions.add(g)

                    for edge_ext in edge_extensions:
                        new_cycle_pattern = new_tree_pattern.apply_edge_extension(edge_ext)
                        debug_message("APPLICO L'ESTENSIONE ARCHI:", "  ,".join([e.__str__() for e in edge_ext]))
                        debug_message(new_cycle_pattern)

                        # If the support of the tree pattern is greater than the cycle pattern
                        # it means that the tree cannot be closed in a cycle for all of his
                        # occurrence in each graph, so it's considered the tree pattern and added to the stack.
                        # Also check if the pattern is not already in the stack, because the same tree can be
                        # considered with more than one edge extension.
                        if (not tree_pattern_added) and (
                                new_tree_pattern.support() > len(graphs_covered_by_edge_extensions)) and (
                                new_tree_pattern.support() > new_cycle_pattern.support()):
                            debug_message(
                                "Il nuovo ciclo non è presente in tutti i grafi, considero anche il vecchio pattern")
                            stack.append(new_tree_pattern)
                            pattern_codes.add(new_pattern_code)
                            tree_pattern_added = True

                        new_cycle_pattern_code = new_cycle_pattern.code()
                        if new_cycle_pattern_code not in pattern_codes:
                            debug_message("Aggiungo il nuovo pattern alla coda")
                            stack.append(new_cycle_pattern)
                            pattern_codes.add(new_cycle_pattern_code)
                        else:
                            debug_message(
                                "BACKTRACKING: L'applicazione dell'estensione degli archi ha fatto trovare un pattern ridondante.")
                            del new_cycle_pattern
                else:
                    debug_message("BACKTRACKING: pattern ridondante")
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
                pattern_mappings = PatternMappings(graphs)
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

    def _init_matcher(self):
        for g in self.db:
            g.init_matcher()
