import copy
from NetworkX.NetworkConfigurator import NetworkConfigurator
from NetworkX.NetworksLoading import NetworksLoading
from CMiner.MultiGraphMatch import MultiGraphMatch
from Graph.Graph import MultiDiGraph
from CMiner.BitMatrix import TargetBitMatrixOptimized, BitMatrixStrategy2
from CMiner.MultiGraphMatch import MultiGraphMatch, Mapping
import time
import shutil
import pandas as pd

with_debugs = False

def print_red(*args, **kwargs):
    if not with_debugs:
        return
    red_text = ' '.join(map(str, args))
    print(f"\033[91m{red_text}\033[0m", **kwargs)

def print_green(*args, **kwargs):
    if not with_debugs:
        return
    green_text = ' '.join(map(str, args))
    print(f"\033[92m{green_text}\033[0m", **kwargs)

def print_yellow(*args, **kwargs):
    if not with_debugs:
        return
    yellow_text = ' '.join(map(str, args))
    print(f"\033[93m{yellow_text}\033[0m", **kwargs)

def print_orange(*args, **kwargs):
    if not with_debugs:
        return
    orange_text = ' '.join(map(str, args))
    print(f"\033[38;5;214m{orange_text}\033[0m", **kwargs)

def print_blue(*args, **kwargs):
    if not with_debugs:
        return
    blue_text = ' '.join(map(str, args))
    print(f"\033[94m{blue_text}\033[0m", **kwargs)

def print_pink(*args, **kwargs):
    if not with_debugs:
        return
    pink_text = ' '.join(map(str, args))
    print(f"\033[95m{pink_text}\033[0m", **kwargs)

class EdgeGroupsFinder:

    """
    Class to find all the edge extensions that are frequent.

    How it works:
        Given a list tuples:  list((edge_labels : list(str), location : dict[DBGraph, list[Mapping]]))
        - Construct a table where the columns are the edge labels and each row contains 0 or 1.
        - The last column contains the location of the edges in the graphs.
        - The table is constructed in such a way that the rows are ordered by the number of 1 in the row.
    """
    def __init__(self, min_support):
        # set the column 'location' as the last column
        self.min_support = min_support
        self.df = pd.DataFrame(columns=['location'])

    def columns(self):
        return list(self.df.columns)

    @staticmethod
    def column_name(label, i):
        """
        Return the column name.
        """
        return label + "_" + str(i)

    @staticmethod
    def label_from_column_name(column_name):
        """
        Return the label from the column name.
        """
        return column_name.rsplit('_', 1)[0]

    @staticmethod
    def parse_edge_labels(edge_labels):
        """
        Parse the edge_labels array. For each edge label add _0. If there are duplicates add _1, _2, ...
        """
        edge_labels_dict = {}
        for i, edge_label in enumerate(edge_labels):
            if edge_label not in edge_labels_dict:
                edge_labels_dict[edge_label] = 1
            else:
                edge_labels_dict[edge_label] += 1
        new_labels = []
        for edge_label, i in edge_labels_dict.items():
            new_labels.extend([EdgeGroupsFinder.column_name(edge_label, n) for n in range(i)])
        return new_labels

    def check_columns(self, edge_labels):
        # CHECK IF THE EDGE LABELS ARE ALREADY PRESENT IN THE COLUMNS
        # NOTE: edge_labels can contain duplicates
        #       e.g.
        #       edge_labels = ['a', 'a', 'b']
        #       columns = ['a_0', 'b_0', 'c_0']
        #       in this case we want to add only 'a_1' because 'a_0','b_0' and 'c_0' is already present in the columns
        columns = self.columns()
        for edge_label in edge_labels:
            if edge_label not in columns:
                self.df[edge_label] = 0

    def compute_new_row(self, edge_labels, location):
        """
        Given a set of edge labels and a location, it returns the new row to add to the dataframe.
        """
        new_row = [0] * len(self.columns())
        cols = self.columns()
        new_row[0] = location
        for l in edge_labels:
            new_row[cols.index(l)] = 1

        return pd.Series(new_row, index=self.df.columns)

    def add_in_order(self, row):
        """
        Add the row in the DataFrame in the correct position.

        The position is determined by the number of 1s in the row.
        The row is added above all the rows which have a number of 1 less than the new row.
        """
        if len(self.df) == 0:
            self.df.loc[0] = row
            return

        new_row_size = sum(row[1:])  # Number of 1s in the new row

        # Trova la posizione corretta in base al numero di 1
        for i in range(len(self.df)):
            row_size = sum(self.df.iloc[i][1:])  # number of 1s in the row
            if row_size < new_row_size:
                # add a row at the end to avoid conflicts
                self.df.loc[len(self.df)] = self.df.iloc[len(self.df) - 1]
                # Shift rows down
                self.df.iloc[i + 1:] = self.df.iloc[i:-1]
                # add new_row
                self.df.loc[i] = row
                return

        # Se non si trova una riga con meno 1s, aggiungi in fondo
        self.df.loc[len(self.df)] = row

    def add(self, edge_labels, location):
        """
        Given a set of edge labels, graphs and mappings, it adds the edge extension to the dataframe.

        Parameters:
        edge_labels (list[str]): edge labels
        graphs (list[DBGraph]): graphs
        mappings (list[Mapping]): mappings
        """
        edge_labels = EdgeGroupsFinder.parse_edge_labels(edge_labels)
        self.check_columns(edge_labels)
        new_row = self.compute_new_row(edge_labels, location)
        self.add_in_order(new_row)

    @staticmethod
    def support(row):
        """
        Return the support of the row.
        """
        return len(row['location'].keys())

    def bitmap(self, row):
        """
        Return the bitmap of the row.
        """
        # each row contains the location of the edges in the graphs, this method returns the bitmap of the row
        # e.g.
        #   row = [{g1: [m1, m2]}, 1, 0, 1]
        #   bitmap = [1, 0, 1]
        return row[1:]

    def is_subset(self, row1, row2):
        """
        Return True if row1 is a subset of row2.
        """
        bitmap1 = self.bitmap(row1)
        bitmap2 = self.bitmap(row2)
        for i in range(len(bitmap1)):
            if bitmap1.iloc[i] > bitmap2.iloc[i]:
                return False
        return True

    @staticmethod
    def extend_location(location1, location2):
        """
        Extend the location of the two rows.
        """
        for g, mappings in location2.items():
            if g in location1:
                location1[g].update(mappings)
            else:
                location1[g] = mappings

    @staticmethod
    def split_into_in_and_out_array(array):
        in_array = []
        out_array = []

        for str in array:
            if str.startswith("in_"):
                in_array.append(str[3:])
            elif str.startswith("out_"):
                out_array.append(str[4:])


        return in_array, out_array

    @staticmethod
    def transform_row_in_extension(row):
        """
        Transform a row in an extension.
        """
        edge_labels = []
        location = {}
        for i, col in enumerate(row.index):
            if i == 0:
                location = row[col]
            elif row[col] == 1:
                edge_labels.append(EdgeGroupsFinder.label_from_column_name(col))
        in_edge_labels, out_edge_labels = EdgeGroupsFinder.split_into_in_and_out_array(edge_labels)
        return Extension(out_edge_labels, in_edge_labels, location)

    def common_columns(self, row1, row2):
        """
        Return the common columns between row1 and row2.
        """
        common = []
        for col in self.columns():
            if row1[col] == 1 and row2[col] == 1:
                common.append(col)
        return common


    def find(self):
        """
        Find all the frequent edge extensions.
        """
        extensions = []

        # other_extensions = {}

        # i := index row to check
        # j := index row to compare with i-th row
        for i in range(len(self.df)):

            row = self.df.iloc[i]

            location = row['location']

            j = i - 1


            while j >= 0:
                row_to_compare = self.df.iloc[j]
                if self.is_subset(row, row_to_compare):
                    # merge the location of the two rows
                    location_row_to_compare = row_to_compare['location']
                    EdgeGroupsFinder.extend_location(location, location_row_to_compare)
                j -= 1

            if EdgeGroupsFinder.support(row) >= self.min_support:
                extensions.append(EdgeGroupsFinder.transform_row_in_extension(row))

        return extensions

    def __str__(self):
        return self.df.__str__()


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

class Extension:

    def __init__(self, out_edge_labels, in_edge_labels, location):
        self.out_edge_labels = out_edge_labels
        self.in_edge_labels = in_edge_labels
        self.location = location

    def graphs(self):
        """
        Return the graphs where the extension is found.
        """
        return list(self.location.keys())

    def mapping(self, graph):
        """
        Return the mapping of the pattern in the graph.
        """
        return self.location[graph]

    def set_graphs(self, graphs):
        """
        Set the graphs where the extension is found.
        """
        self.location = {g: {} for g in graphs}


class NodeExtension(Extension):

    def __init__(self, pattern_node_id, node_labels, out_edge_labels, in_edge_labels, location):
        super().__init__(out_edge_labels, in_edge_labels, location)
        self.pattern_node_id = pattern_node_id
        self.node_labels = node_labels

    def target_node_id(self, graph, _map):
        """
        Return the target node id from which the extension is found.
        """
        for m, target_node_id in self.location[graph]:
            if _map == m:
                return target_node_id
        return None

    def __str__(self):
        g_names = sorted([g.name for g in self.location.keys()])
        graphs = ", ".join(g_names)
        return f"NodeExt: (\n   PatternNodeId: {self.pattern_node_id}\n   NewNodeLabels: {self.node_labels}\n   OutEdgeLabels: {self.out_edge_labels}\n   InEdgeLabels: {self.in_edge_labels}\n   Location: {graphs}\n)"

class EdgeExtension(Extension):

    def __init__(self, pattern_node_id_src, pattern_node_id_dst, out_edge_labels, in_edge_labels, location):
        super().__init__(out_edge_labels, in_edge_labels, location)
        self.pattern_node_id_src = pattern_node_id_src
        self.pattern_node_id_dst = pattern_node_id_dst

    def __str__(self):
        g_names = sorted([g.name for g in self.location.keys()])
        graphs = ", ".join(g_names)
        return f"EdgeExt: (\n   PatternNodeIdSrc: {self.pattern_node_id_src}\n   PatternNodeIdDst: {self.pattern_node_id_dst}\n   OutEdgeLabels: {self.out_edge_labels}\n   InEdgeLabels: {self.in_edge_labels}\n   Location: {graphs}\n)"

    def __copy__(self):
        return EdgeExtension(
            self.pattern_node_id_src,
            self.pattern_node_id_dst,
            self.out_edge_labels,
            self.in_edge_labels,
            self.location
        )

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

    def __str__(self):
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

class NodeExtensionManager:

    def __init__(self, support):
        self.min_support = support
        self.extensions = {}

    def add(self, pattern_node_id, target_node_id, neigh_target_node_id, db_graph, _map):
        """
        Add an extension to the manager

        Parameters:
            pattern_node_id (int): the id of the node in the pattern that is extended
            target_node_id (int): the id of the node mapped with pattern_node_id
            neigh_target_node_id (int): neighbor of target_node_id
            db_graph (DBGraph): the graph where the extension is found
            _map (Mapping): the mapping of the pattern in the db_graph
        """
        edge_labels = []
        for label in db_graph.get_edge_labels_with_duplicate(target_node_id, neigh_target_node_id):
            edge_labels.append(NodeExtensionManager.orientation_code(label, True))
        for label in db_graph.get_edge_labels_with_duplicate(neigh_target_node_id, target_node_id):
            edge_labels.append(NodeExtensionManager.orientation_code(label, False))

        neigh_target_node_labels = db_graph.get_node_labels(neigh_target_node_id)
        edge_labels = sorted(edge_labels)

        neigh_target_node_labels_code = " ".join(neigh_target_node_labels)
        target_edge_labels_code = " ".join(edge_labels)

        ext_code = (pattern_node_id, neigh_target_node_labels_code, target_edge_labels_code)

        if ext_code not in self.extensions:
            self.extensions[ext_code] = {}
        if db_graph not in self.extensions[ext_code]:
            self.extensions[ext_code][db_graph] = []
        self.extensions[ext_code][db_graph].append((_map, neigh_target_node_id))



    def frequent_extensions(self) -> list['NodeExtension']:
        """
        Return a list of NodeExtensions that if applied to the pattern, it still remains frequent.
        """
        frequent_extensions = []

        edge_group_finders = {}

        for (pattern_node_id, node_labels_code, target_edge_labels_code), db_graphs in self.extensions.items():

            # use the finder code to identify the finder
            finder_code = (pattern_node_id, node_labels_code)
            # instantiate the finder if it is not present
            if finder_code not in edge_group_finders:
                edge_group_finders[finder_code] = EdgeGroupsFinder(self.min_support)

            # create the location dictionary
            location = {}
            for g in db_graphs:
                location[g] = set(db_graphs[g])

            # select the correct finder and add the edge extension
            edge_group_finder = edge_group_finders[finder_code]
            edge_group_finder.add(target_edge_labels_code.split(" "), location)

        # save all frequent extensions
        for (pattern_node_id, node_labels_code), edge_group_finder in edge_group_finders.items():
            ext = edge_group_finder.find()
            for e in ext:
                frequent_extensions.append(NodeExtension(pattern_node_id, node_labels_code.split(" "), e.out_edge_labels, e.in_edge_labels, e.location))

        return frequent_extensions


    @staticmethod
    def orientation_code(label, outgoing):
        """
        Return the code of the orientation of the edge.

        E.g.
            src_node -- edge_label --> dst_node

            edge_label became 'out_edge_label' if outgoing is True
            edge_label became 'in_edge_label' if outgoing is False
        """
        return ("out_" if outgoing else "in_") + label

class EdgeExtensionManager:

    def __init__(self, support):
        self.min_support = support
        self.extensions = {}

    def add(self, pattern_node_src, pattern_node_dest, labels, db_graph, _map):
        """
        Add an extension to the manager.
        """
        # edge_labels = []
        # target_src_node_id = _map.node(pattern_node_src)
        # target_dst_node_id = _map.node(pattern_node_dest)

        # for label in db_graph.get_edge_labels_with_duplicate(target_src_node_id, target_dst_node_id):
        #     edge_labels.append(NodeExtensionManager.orientation_code(label, True))
        # ??????????????
        # if with_in_labels:
        #     for label in db_graph.get_edge_labels_with_duplicate(target_dst_node_id, target_src_node_id):
        #         edge_labels.append(NodeExtensionManager.orientation_code(label, False))
        # ??????????????

        # edge_labels = sorted(edge_labels)

        # target_edge_labels_code = " ".join(edge_labels)

        # extension_code = (pattern_node_src, pattern_node_dest, target_edge_labels_code)

        target_edge_labels_code = " ".join(sorted([NodeExtensionManager.orientation_code(label, True) for label in labels]))
        extension_code = (pattern_node_src, pattern_node_dest, target_edge_labels_code)


        if extension_code not in self.extensions:
            self.extensions[extension_code] = {}
        if db_graph not in self.extensions[extension_code]:
            self.extensions[extension_code][db_graph] = []
        self.extensions[extension_code][db_graph].append(_map)

    def frequent_extensions(self) -> list['EdgeExtension']:
        """
        Return a list of Extensions that are frequent
        """
        frequent_extensions = []

        edge_group_finders = {}

        for (pattern_node_src, pattern_node_dest, target_edge_labels_code), db_graphs in self.extensions.items():

            # use the finder code to identify the finder
            finder_code = (pattern_node_src, pattern_node_dest)
            # instantiate the finder if it is not present
            if finder_code not in edge_group_finders:
                edge_group_finders[finder_code] = EdgeGroupsFinder(self.min_support)

            # create the location dictionary
            location = {}
            for g in db_graphs:
                location[g] = set(db_graphs[g])

            # select the correct finder and add the edge extension
            edge_group_finder = edge_group_finders[finder_code]
            edge_group_finder.add(target_edge_labels_code.split(" "), location)

        # save all frequent extensions
        for (pattern_node_src, pattern_node_dest), edge_group_finder in edge_group_finders.items():
            ext = edge_group_finder.find()
            for e in ext:
                frequent_extensions.append(EdgeExtension(pattern_node_src, pattern_node_dest, e.out_edge_labels, e.in_edge_labels, e.location))

        return frequent_extensions


class Pattern(MultiDiGraph):

    def __init__(self, pattern_mappings, extended_pattern: 'Pattern' = None,
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

    def find_node_extensions(self, min_support) -> list[NodeExtension]:
        """
        Generate all possible node extension that if applied to the pattern, it still remains frequent.
        """

        print_orange("--- Find node extensions ---")
        print_orange(self)

        extension_manager = NodeExtensionManager(min_support)
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
                    for neigh in g.all_neighbors(node_db).difference(mapped_target_nodes):
                        extension_manager.add(node_p, node_db, neigh, g, _map)

        extensions = extension_manager.frequent_extensions()
        # DELETE THIS ONLY FOR TESTING
        extensions = sorted(extensions, key=lambda x: x.__str__())

        return extensions


    def find_edge_extensions(self, min_support) -> list[list[EdgeExtension]]:

        print_orange("--- Find edge extensions ---")
        print_orange(self)

        if len(self.nodes()) < 3:
            # if the pattern has less than 3 nodes, it is not possible to find edge extensions
            return []


        extension_manager = EdgeExtensionManager(min_support)

        for g in self.graphs():
            for _map in self.pattern_mappings.mappings(g):
                # subgraph of the projected pattern (include also edges not mapped with the patten)
                mapped_pattern_complete_graph = g.subgraph(_map.nodes())
                print_blue(g.name, mapped_pattern_complete_graph)
                mapped_pattern_complete_graph_edges = set(mapped_pattern_complete_graph.edges(keys=True))
                mapped_pattern_edges = set(_map.get_target_edges())
                candidate_edges = set()


                print_blue("Mapped pattern complete graph edges", mapped_pattern_complete_graph_edges)
                print_blue("Mapped pattern edges", mapped_pattern_edges)


                # for src, dst, _ in mapped_pattern_complete_graph_edges:
                #     if all(src != s or dst != d for s, d, _ in mapped_pattern_edges):
                #         candidate_edges.add((src, dst))

                for src, dst, key in mapped_pattern_complete_graph_edges:
                    skip = False
                    for s, d, k in mapped_pattern_edges:
                        if src == s and dst == d:
                            # remove i-th element from the list
                            ss, dd, kk = s, d, k
                            mapped_pattern_edges.remove((ss, dd, kk))
                            skip = True
                            break
                    if skip:
                        continue
                    candidate_edges.add((src, dst, key, g.get_edge_label((src, dst, key))))

                print_blue("Candidate edges", candidate_edges)

                groups = {}

                inverse_map = _map.inverse()
                for src, dst, key, lab in candidate_edges:
                    pattern_node_src = inverse_map.node(src)
                    pattern_node_dest = inverse_map.node(dst)
                    code = (pattern_node_src, pattern_node_dest)
                    if code not in groups:
                        groups[code] = []
                    groups[code].append(lab)

                print_blue("Groups", groups)

                for (src, dst), labels in groups.items():
                    extension_manager.add(src, dst, labels, g, _map)

        extensions = extension_manager.frequent_extensions()

        # DELETE THIS ONLY FOR TESTING
        extensions = sorted(extensions, key=lambda x: x.__str__())

        # DELETE THIS ONLY FOR TESTING
        print_green("Edge extensions before grouping")
        for e in extensions:
            print_green(e)

        if len(extensions) == 0:
            return []

        graphs = sorted(self.graphs(), key=lambda x: x.get_name())
        extension_matrix = [[0 for _ in range(len(graphs))] for _ in range(len(extensions))]
        for i, ext in enumerate(extensions):
            for j, g in enumerate(graphs):
                if g in ext.graphs():
                    extension_matrix[i][j] = 1

        # group row by row
        matrix_indices_grouped = {}
        for i, row in enumerate(extension_matrix):
            row_code = "".join(map(str, row))
            if row_code not in matrix_indices_grouped:
                matrix_indices_grouped[row_code] = []
            matrix_indices_grouped[row_code].append(i)

        print_pink("Matrix indices grouped")
        print_pink(matrix_indices_grouped)


        groups = []
        for row_code, indices in matrix_indices_grouped.items():
            columns_to_select = [i for i, v in enumerate(row_code) if v == "1"]
            group = []
            for i, ext in enumerate(extensions):
                skip = False
                if all(extension_matrix[i][j] == 1 for j in columns_to_select):
                    for e in group:
                        if ext.pattern_node_id_src == e.pattern_node_id_src and ext.pattern_node_id_dst == e.pattern_node_id_dst:
                            print_red(ext)
                            print_red(e)
                            skip = True
                            break
                    if skip:
                        continue
                    ext_copy = ext.__copy__()
                    new_location = {v: k for v, k in ext.location.items() if any(v == graphs[j] for j in columns_to_select)}
                    ext_copy.location = new_location
                    group.append(ext_copy)
            groups.append(group)

        return groups

    def apply_node_extension(self, extension: NodeExtension) -> 'Pattern':
        """
        Apply the node extension to the pattern.
        """
        print_yellow("--- Apply node extension ---")
        print_yellow(extension)

        # Object to keep track of the new pattern mappings
        new_pattern_mappings = PatternMappings()
        # The id of the previous pattern node that is extended
        pattern_node_id = extension.pattern_node_id

        # Apply extension to the pattern (add node and edges)
        new_pattern = Pattern(extended_pattern=self, pattern_mappings=new_pattern_mappings)
        new_pattern_new_node_id = int(len(new_pattern.nodes())) + 1
        new_pattern.add_node(new_pattern_new_node_id, labels=extension.node_labels)

        for lab in extension.in_edge_labels:
            new_pattern.add_edge(new_pattern_new_node_id, pattern_node_id, type=lab)
        for lab in extension.out_edge_labels:
            new_pattern.add_edge(pattern_node_id, new_pattern_new_node_id, type=lab)

        # Update the pattern mappings
        for target in extension.graphs():
            new_mappings = []
            for target_map in self.pattern_mappings.mappings(target):

                # # Check if the extension is applicable by verifying if target_map is in extension_node_ids[target]
                # # (it means that the extension is applicable because it let us discover a new extension)
                # if target_map not in extension.mapping(target):
                #     continue

                target_node_id = extension.target_node_id(target, target_map)

                # when trying to extend the pattern Pn (pattern with n nodes), there can be some mappings of Pn
                # that are not extended because the extension is not applicable.
                if target_node_id is None:
                    continue


                # node mapping
                node_mapping = {new_pattern_new_node_id: target_node_id}
                # edge mapping
                edge_mapping = {}

                new_key = 0
                prev_lab = None
                prev_keys = []
                for lab in sorted(extension.in_edge_labels):
                    new_pattern_edge = (pattern_node_id, new_pattern_new_node_id, new_key)
                    target_edge_src = target_node_id
                    target_edge_dest= target_map.nodes_mapping()[extension.pattern_node_id]
                    if prev_lab != lab:
                        prev_keys = target.edge_keys_by_type(target_edge_src, target_edge_dest, lab)
                        prev_lab = lab
                    target_edge = (target_edge_src, target_edge_dest, prev_keys.pop(0))
                    edge_mapping[new_pattern_edge] = target_edge
                    new_key += 1
                prev_lab = None
                prev_keys = []
                for lab in sorted(extension.out_edge_labels):
                    new_pattern_edge = (new_pattern_new_node_id, pattern_node_id, new_key)
                    target_edge_src = target_map.nodes_mapping()[extension.pattern_node_id]
                    target_edge_dest = target_node_id
                    if prev_lab != lab:
                        prev_keys = target.edge_keys_by_type(target_edge_src, target_edge_dest, lab)
                        prev_lab = lab
                    target_edge = (target_edge_src, target_edge_dest, prev_keys.pop(0))
                    edge_mapping[new_pattern_edge] = target_edge
                    new_key += 1

                new_mapping = Mapping(node_mapping=node_mapping, edge_mapping=edge_mapping, extended_mapping=target_map)
                new_mappings.append(new_mapping)

            new_pattern_mappings.set_mapping(target, new_mappings)
        return new_pattern

    def apply_edge_extension(self, extensions: list[EdgeExtension]) -> 'Pattern':
        """
        Apply the edge extension to the pattern.
        """

        print_yellow("--- Apply edge extension ---")
        for ext in extensions:
            print_yellow(ext)


        db_graphs = extensions[0].graphs()
        new_pattern_mappings = PatternMappings()

        # Apply extension to the pattern (add edges)
        new_pattern = Pattern(extended_pattern=self, pattern_mappings=new_pattern_mappings)

        for ext in extensions:
            for lab in ext.in_edge_labels:
                new_pattern.add_edge(ext.pattern_node_id_dst, ext.pattern_node_id_src, type=lab)
            for lab in ext.out_edge_labels:
                new_pattern.add_edge(ext.pattern_node_id_src, ext.pattern_node_id_dst, type=lab)

        # Update the pattern mappings
        for target in db_graphs:

            new_mappings = []

            for target_map in self.pattern_mappings.mappings(target):

                try:
                    if any(target_map not in ext.mapping(target) for ext in extensions):
                        continue
                except:
                    # target could not be associated to any mapping in the extension
                    continue

                new_mapping = Mapping(extended_mapping=target_map)

                for extension in extensions:

                    target_edge_src = target_map.nodes_mapping()[extension.pattern_node_id_src]
                    target_edge_dest = target_map.nodes_mapping()[extension.pattern_node_id_dst]
                    new_key = 0
                    prev_lab = None
                    prev_keys = []
                    for lab in sorted(extension.out_edge_labels):
                        if prev_lab != lab:
                            prev_keys = target.edge_keys_by_type(target_edge_src, target_edge_dest, lab)
                            prev_lab = lab
                        if len(prev_keys) == 0:
                            continue
                        target_edge = (target_edge_src, target_edge_dest, prev_keys.pop(0))
                        new_mapping.set_edge((extension.pattern_node_id_dst, extension.pattern_node_id_src, new_key),
                                             target_edge)
                        new_key += 1
                    new_key = 0
                    prev_lab = None
                    prev_keys = []
                    for lab in sorted(extension.in_edge_labels):
                        if prev_lab != lab:
                            prev_keys = target.edge_keys_by_type(target_edge_src, target_edge_dest, lab)
                            prev_lab = lab
                        if len(prev_keys) == 0:
                            continue
                        target_edge = (target_edge_src, target_edge_dest, prev_keys.pop(0))
                        new_mapping.set_edge((extension.pattern_node_id_src, extension.pattern_node_id_dst, new_key),
                                             target_edge)
                        new_key += 1

                new_mappings.append(new_mapping)
            new_pattern_mappings.set_mapping(target, new_mappings)

        return new_pattern


    def __str__(self, show_mappings=False):
        global count_pattern
        output = ""
        # graph info
        code = self.code()
        output += f"t # {count_pattern} {code}\n"
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
            # DELETE THIS ONLY FOR TESTING
            frequencies = sorted(frequencies)
            #
            output += "x " + " ".join(frequencies) + "\n"
        output += "----------" + "\n"

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

    def output(self, pattern):

        output_file = None
        if self.output_path is not None:
            output_file = open(self.output_path, "a")

        if len(pattern.nodes()) >= self._min_nodes:
            if self.output_path is not None:
                print("New solution")
                print(pattern.__str__(self.show_mappings), file=output_file)
            else:
                print(pattern.__str__(self.show_mappings))

        if output_file is not None:
            output_file.close()

    def mine(self):
        self._read_graphs_from_file()
        self._parse_support()

        pattern_codes = set()

        # Stack for DFS
        stack = self.find_start_patterns()
        # DELETE THIS ONLY FOR TESTING
        stack = sorted(stack, key=lambda x: x.__str__())

        for p in stack:
            self.output(p)

        while len(stack) > 0:
            pattern_to_extend = stack.pop()

            # Check if the pattern is already at the max number of nodes
            if len(pattern_to_extend.nodes()) >= self._max_nodes:
                del pattern_to_extend
                continue

            print_red("--- Working on ---")
            print_red(pattern_to_extend)

            # Find extensions
            node_extensions = pattern_to_extend.find_node_extensions(self.min_support)

            if len(node_extensions) == 0:
                print_orange("No node extensions found")
                del pattern_to_extend
                # Backtracking occurs when no more extensions are found
                continue

            for node_ext in node_extensions:

                new_tree_pattern = pattern_to_extend.apply_node_extension(node_ext)
                new_pattern_code = new_tree_pattern.code()

                # Ensure no duplicate patterns are processed
                if new_pattern_code not in pattern_codes:

                    tree_pattern_added = False

                    edge_extensions = new_tree_pattern.find_edge_extensions(self.min_support)

                    # edge_extensions = []

                    # If no edge extensions are found, add the tree pattern to the stack
                    if len(edge_extensions) == 0:
                        print_orange("No edge extensions found")
                        self.output(new_tree_pattern)
                        stack.append(new_tree_pattern)
                        pattern_codes.add(new_pattern_code)
                        continue

                    # ????????????????
                    graphs_covered_by_edge_extensions = set()
                    for edge_ext in edge_extensions:
                        for g in edge_ext[0].graphs():
                            graphs_covered_by_edge_extensions.add(g)
                    # ????????????????



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
                            self.output(new_tree_pattern)
                            stack.append(new_tree_pattern)
                            pattern_codes.add(new_pattern_code)
                            tree_pattern_added = True

                        new_cycle_pattern_code = new_cycle_pattern.code()
                        if new_cycle_pattern_code not in pattern_codes:
                            self.output(new_cycle_pattern)
                            stack.append(new_cycle_pattern)
                            pattern_codes.add(new_cycle_pattern_code)
                        else:
                            del new_cycle_pattern
                else:
                    del new_tree_pattern


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

