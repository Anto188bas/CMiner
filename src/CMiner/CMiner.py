import os
from src.NetworkX.NetworkConfigurator import NetworkConfigurator
from src.NetworkX.NetworksLoading import NetworksLoading
from src.CMiner.MultiGraphMatch import MultiGraphMatch
from src.Graph.Graph import MultiDiGraph
from src.CMiner.BitMatrix import TargetBitMatrixOptimized, BitMatrixStrategy2
import copy
import time

# TO-DO: handle multiple edges map in _apply_mappings

def debug_message(*args, **kwargs):
    if False:
        print(*args, **kwargs)


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
                extensions_node_id = extensions_node_ids[(pattern_node_id, target_node_labels, target_edge_labels)]
                target_node_labels = target_node_labels.split(" ")
                target_edge_labels = target_edge_labels.split(" ")
                ext = Extension(pattern_node_id, target_node_labels, target_edge_labels, is_outgoing, db_graphs,
                                extensions_node_id)
                debug_message("                - Extension", ext)
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
        self.extension_node_ids = extension_node_ids

    def __str__(self):
        if self.is_outgoing:
            output = f"({self.pattern_node_id} -- {self.target_edge_labels} --> {self.target_node_labels}) "
        else:
            output = f"({self.pattern_node_id} <-- {self.target_edge_labels} -- {self.target_node_labels}) "
        output += str(self.extension_node_ids)
        return output


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

    def get_target_nodes(self):
        return self.node_mapping.values()

    def get_target_edges(self):
        return self.edge_mapping.values()

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


class Pattern(MultiDiGraph):

    def __init__(self, pattern_mappings, extended_pattern: 'Pattern' = None, **attr):

        if extended_pattern is not None:
            # Copia i nodi e gli archi dal pattern esteso
            super().__init__(extended_pattern, **attr)
        else:
            super().__init__(**attr)

        self.extended_pattern = extended_pattern
        self.pattern_mappings = pattern_mappings

    def extend(self, support) -> list['Pattern']:

        extension_manager = ExtensionManager(support)
        new_patterns = []
        extensions = self._find_extensions(extension_manager)
        debug_message("        - Found", len(extensions), "extensions")
        for ext in extensions:
            new_patterns.append(self._apply_extension(ext))

        new_patterns = remove_redundant_patterns(new_patterns)
        return new_patterns

    def _apply_extension(self, extension) -> 'Pattern':
        debug_message("            - Applying extension", extension)
        # Object to keep track of the new pattern mappings
        new_pattern_mappings = PatternMappings(extension.db_graphs)
        # The id of the previous pattern node that is extended
        pattern_node_id = extension.pattern_node_id

        # Apply extension to the pattern (add node and edges)
        new_pattern = Pattern(extended_pattern=self, pattern_mappings=new_pattern_mappings)
        new_pattern_new_node_id = max(new_pattern.nodes()) + 1
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
                        target_edge = (target_map.nodes_mapping()[pattern_node_id], target_node_id, 0)
                    else:
                        new_pattern_edge = (new_pattern_new_node_id, pattern_node_id, 0)
                        target_edge = (target_node_id, target_map.nodes_mapping()[pattern_node_id], 0)
                    edge_mapping = {new_pattern_edge: target_edge}
                    new_mapping = Mapping(node_mapping=node_mapping, edge_mapping=edge_mapping, extended_mapping=target_map)
                    new_mappings.append(new_mapping)
            new_pattern_mappings.set_mapping(target, new_mappings)
        return new_pattern

    # def find_cycles(self, support) -> list['Pattern']:
    #     candidate_edges_to_extend_count = {}
    #
    #     inverse_mappings = {}
    #     for g in self.pattern_mappings.get_graphs():
    #         for _map in self.pattern_mappings.get_mappings(g):
    #             if g not in inverse_mappings:
    #                 inverse_mappings[g] = {}
    #             inverse_mappings[g][_map] = {v: k for k, v in _map.nodes_mapping().items()}
    #
    #     # Processa ogni grafo una volta e calcola il sottografo solo una volta
    #     for g in self.pattern_mappings.get_graphs():
    #         for _map in self.pattern_mappings.get_mappings(g):
    #             mapped_nodes = set(_map.get_target_nodes())
    #             mapped_edges = set(_map.get_target_edges())
    #
    #             # Ottieni il sottografo solo una volta per ogni grafo
    #             projection = g.subgraph(mapped_nodes)
    #             candidate_target_edges_to_extend = set(projection.edges(keys=True)).difference(mapped_edges)
    #
    #             # Itera solo sui bordi non mappati
    #             for target_edge in candidate_target_edges_to_extend:
    #                 target_edge_label = g.get_edge_label(target_edge)
    #                 target_edge_src, target_edge_dst = target_edge[0], target_edge[1]
    #
    #                 # Trova rapidamente i nodi nel pattern usando la mappatura inversa
    #                 pattern_node_src = inverse_mappings[g][_map].get(target_edge_src)
    #                 pattern_node_dst = inverse_mappings[g][_map].get(target_edge_dst)
    #
    #                 if pattern_node_src is None or pattern_node_dst is None:
    #                     continue  # Se non sono mappati, salta
    #
    #                 ext = (pattern_node_src, pattern_node_dst, target_edge_label)
    #
    #                 if ext not in candidate_edges_to_extend_count:
    #                     candidate_edges_to_extend_count[ext] = set()
    #
    #                 # Aggiungi il grafo al set di grafi che contiene questo bordo
    #                 candidate_edges_to_extend_count[ext].add(g)
    #
    #     # Filtro per gli archi frequenti
    #     frequent_edges = {edge: graphs for edge, graphs in candidate_edges_to_extend_count.items() if
    #                       len(graphs) >= support}
    #
    #     # Estendi il pattern solo per gli archi frequenti
    #     for edge, graphs in frequent_edges.items():
    #         self.add_edge(edge[0], edge[1], type=edge[2])
    #         self.pattern_mappings.set_graphs(graphs)

    def find_cycles(self, support) -> list['Pattern']:
        print("Find cycles")
        # debug_message("Find cycles")
        # debug_message(self)
        # debug_message("Find cycles for pattern")
        # debug_message(self)
        candidate_edges_to_extend_count = {}
        for g in self.pattern_mappings.get_graphs():
            for _map in self.pattern_mappings.get_mappings(g):
                # all nodes in g that are mapped with the pattern
                mapped_nodes = set(_map.get_target_nodes())
                # all edges in g that are mapped with the pattern
                mapped_edges = set(_map.get_target_edges())
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
        if len(frequent_edges.items()) > 0:
            print("trovato ciclo")
        for edge, graphs in frequent_edges.items():
            self.add_edge(edge[0], edge[1], type=edge[2])
            self.pattern_mappings.set_graphs(graphs)
        # debug_message("New pattern")
        # debug_message(self)

    def _find_extensions(self, extension_manager: ExtensionManager) -> list[Extension]:
        """
        Generate all possible extension that if applied to the pattern, it still remains frequent.

        :param extension_manager: Object that manage the extensions
        :return: List of all possible extensions
        """
        debug_message("        - Searching extensions in", len(self.pattern_mappings.get_graphs()), "graphs")
        # for all graph in the database that contains the current extension
        for g in self.pattern_mappings.get_graphs():
            # obtain where the current extension is located in the graph
            mappings = self.pattern_mappings.get_mappings(g)
            debug_message("            - Checking in", len(mappings), "mappings for graph", g.get_name())
            # For each map we know one place where the extension is located in the graph.
            # We search all nodes that are neighbors of the current pattern and create a new extension.
            for _map in mappings:
                # get where the nodes of the extension are located in the DB graph
                node_mapping = _map.nodes_mapping()
                debug_message("                - Node Mapping", node_mapping)
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



        if show_mappings:
            for g in self.pattern_mappings.get_graphs():
                # graph_name frequency
                output += g.get_name() + " " + str(len(self.pattern_mappings.get_mappings(g))) + " "
                # ({nodes map}, {edges map}), ...
                output += " ".join(
                    [str(_map) for _map in self.pattern_mappings.get_mappings(g)])
                output += "\n"
        else:
            #frequency info
            frequencies = ["(" + g.get_name() + ", " + str(len(self.pattern_mappings.get_mappings(g))) + ")" for g in self.pattern_mappings.get_graphs()]
            output += "x " + " ".join(frequencies) + "\n"
        output += "-------------\n"
        count_pattern += 1
        return output


class CMiner:

    def __init__(self,
                 graph_db_path,
                 min_num_nodes=1,
                 max_num_nodes=float('inf'),
                 support=0.5,
                 start_pattern=None, # implement
                 show_mappings=False,
                 output_path=None,
                 approach='dfs'
                 ):
        self.graph_db_path = graph_db_path
        self._min_num_nodes = min_num_nodes
        self._max_num_nodes = max_num_nodes
        self.support = support
        self._start_pattern = start_pattern
        self.db = []
        self.show_mappings = show_mappings
        self.output_path = output_path
        self.approach = approach

    def mine(self):
        if self.approach.lower() == 'dfs':
            self.dfs_mine()
        elif self.approach.lower() == 'bfs':
            self.bfs_mine()
        else:
            raise Exception("Invalid approach [dfs, bfs]")

    def dfs_mine(self):
        print("DFS Mining")
        self._read_graphs_from_file()
        self._parse_support()
        pattern_codes = set()

        # Stack for DFS
        stack = self._mine_1node_patterns()

        # Open the file initially
        output_file = None
        if self.output_path is not None:
            output_file = open(self.output_path, "a")

        while len(stack) > 0:
            pattern_to_extend = stack.pop()

            # Print pattern to console and file if it meets the min node requirement
            if len(pattern_to_extend.nodes()) >= self._min_num_nodes:
                if self.output_path is not None:
                    print(pattern_to_extend.__str__(self.show_mappings), file=output_file)
                else:
                    print(pattern_to_extend.__str__(self.show_mappings))

            # Check if the pattern is already at the max number of nodes
            if len(pattern_to_extend.nodes()) == self._max_num_nodes:
                del pattern_to_extend
                continue 

            # Find extensions
            extensions = pattern_to_extend._find_extensions(ExtensionManager(self.support))
            if len(extensions) == 0:
                del pattern_to_extend
                # Backtracking occurs when no more extensions are found
                if self.output_path is not None:
                    output_file.close()  # Close the file on backtrack
                    output_file = open(self.output_path, "a")  # Reopen the file
                continue

            for ext in extensions:
                new_pattern = pattern_to_extend._apply_extension(ext)
                new_pattern_code = new_pattern.code()

                # Ensure no duplicate patterns are processed
                if new_pattern_code not in pattern_codes:
                    stack.append(new_pattern)
                    pattern_codes.add(new_pattern_code)
                    # # Find cycles
                    # new_pattern.find_cycles(self.support)
                    # new_pattern_code = new_pattern.code()
                    # if new_pattern_code not in pattern_codes:
                    #     stack.append(new_pattern)
                    #     pattern_codes.add(new_pattern_code)

        # Close the file if it was opened
        if output_file is not None:
            output_file.close()

    def bfs_mine(self):
        print("BFS Mining")
        self._read_graphs_from_file()
        self._parse_support()
        # define the number of nodes to start mining
        curr_nodes = 1
        # count the number of patterns found
        pattern_count = 0
        while curr_nodes <= self._max_num_nodes:
            # mine patterns
            patterns = self._mine_patterns(curr_nodes)

            if len(patterns) == 0:
                break

            if self.output_path is not None:
                output_file = open(self.output_path, "a")

            if curr_nodes >= self._min_num_nodes:
                # print results only if the current number of nodes is at least the min number of nodes
                for pattern in patterns:
                    if self.output_path is not None:
                        print(pattern.__str__(self.show_mappings), file=output_file)
                    else:
                        print(pattern.__str__(self.show_mappings))
                    pattern_count += 1
            if self.output_path is not None:
                output_file.close()
            # increment the number of nodes
            curr_nodes += 1

    def _mine_patterns(self, num_nodes) -> list[Pattern]:
        print("Mine patterns with", num_nodes, "nodes")
        if num_nodes == 1:
            patterns = self._mine_1node_patterns()
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

        # update the mappings
        patterns = []
        for sorted_labels_str, graphs in counter.items():
            if len(graphs) >= self.support:
                pattern_mappings = PatternMappings(graphs)
                p = Pattern(pattern_mappings)
                p.add_node(0, labels=sorted_labels_str.split(" "))
                for g in graphs:
                    p.pattern_mappings.set_mapping(g, [Mapping(node_mapping={0: node}) for node in g.nodes() if
                                                       g.get_node_labels(node) == sorted_labels_str.split(" ")])
                patterns.append(p)
        return patterns

    def _read_graphs_from_file(self):
        type_file = self.graph_db_path.split('.')[-1]
        configurator = NetworkConfigurator(self.graph_db_path, type_file)
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
