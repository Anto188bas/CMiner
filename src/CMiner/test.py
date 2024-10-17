import pandas as pd
import copy

class Extension:

    def __init__(self, edge_labels, location):
        self.edge_labels = edge_labels
        self.location = location

    def __str__(self):
        return str(self.edge_labels) + " " + str(self.location)

class NodeExtension(Extension):

    def __init__(self, src_id, node_labels, edge_labels, location, is_outgoing):
        super().__init__(edge_labels, location)
        self.src_id = src_id
        self.node_labels = node_labels
        self.is_outgoing = is_outgoing

class EdgeExtension(Extension):

    def __init__(self, src_id, dst_id, edge_labels, location):
        super().__init__(edge_labels, location)
        self.src_id = src_id
        self.dst_id = dst_id


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
        return column_name.split('_')[0]

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
    def transform_row_in_extension(row):
        """
        Transform a row in an extension.
        """
        print(type(row))
        edge_labels = []
        location = {}
        for i, col in enumerate(row.index):
            if i == 0:
                location = row[col]
            elif row[col] == 1:
                edge_labels.append(EdgeGroupsFinder.label_from_column_name(col))
        return Extension(edge_labels, location)

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
                # else:
                #
                #     common_columns = self.common_columns(row, row_to_compare)
                #     if len(common_columns) > 0:
                #         # if there are two rows have common columns, it means that there are possible extensions
                #
                #         # obtain the bitmap representing the common columns
                #         common_columns_code = "__".join(sorted(common_columns))
                #         if common_columns_code not in other_extensions:
                #             new_location = copy.deepcopy(location)
                #             EdgeGroupsFinder.extend_location(new_location, row_to_compare['location'])
                #             other_extensions[common_columns_code] = new_location
                #         else:
                #             EdgeGroupsFinder.extend_location(other_extensions[common_columns_code], row_to_compare['location'])

                j -= 1

            if EdgeGroupsFinder.support(row) >= self.min_support:
                extensions.append(EdgeGroupsFinder.transform_row_in_extension(row))

        # for common_columns_code, locations in other_extensions.items():
        #     if len(locations) >= self.min_support:
        #         extensions.append(EdgeGroupsFinder.transform_row_in_extension(self.compute_new_row(common_columns_code.split('__'), locations)))

        return extensions

    def __str__(self):
        return self.df.__str__()



eg = EdgeGroupsFinder(2)

# eg.add(['r', 'r', 'r', 'b', 'b'], {'g1': {'a'}})
# eg.add(['r', 'r', 'r'], {'g3': {'h'}})
# eg.add(['b', 'b', 'v'], {'g3': {'i'}})
# eg.add(['b', 'v'], {'g2': {'f'}})
# eg.add(['r', 'r'], {'g2': {'e'}})
# eg.add(['b', 'b'], {'g1': {'b'}})

eg.add(['r', 'r', 'r', 'b', 'b'], {'g1': {'a'}})
eg.add(['r', 'r', 'r'], {'g3': {'h'}})
eg.add(['b', 'b', 'p'], {'g3': {'d'}})
eg.add(['b', 'p'], {'g2': {'f'}})
eg.add(['r', 'r'], {'g2': {'e'}})
eg.add(['b', 'b'], {'g1': {'b'}})




print()
print(eg)
print()




exts = eg.find()

for ext in exts:
    print(ext)
