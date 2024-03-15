from abc import ABC, abstractmethod
from tabulate import tabulate

# TO-DO: Find an optimal way to discard edges already 
#        computed in the compute method [O(log n)?]
# TO-DO: Find candidates in a different way. 
#        Represent BitMatrix as a BTree?
# TO-DO: Keep lazy computing?
# DONE:  Write better the code for showing the matrix
# DONE:  Optimization on the QueryBitMatrix computation.
#        After computing bitmap for edge (a, b), bitmap
#        For edge (b, a) can the computed with two 
#        permutaion on the bitmap for (a, b)


class BitMatrix(ABC):
    
    """ Abstract class for describing a BitMatrix
    
    :param graph               Graph in which the BitMatrix is computed
    :type  graph               Multigraph
    :param bit_matrix_strategy Datastructure for the rows of the matrix
    :type  bit_matrix_strategy BitMatrixStrategy
    """
    def __init__(self, graph, bit_matrix_strategy):
        bit_matrix_strategy.set_graph(graph)
        self.bit_matrix_strategy = bit_matrix_strategy
        self.graph = graph
        # BitMatrix (list of bitmaps)
        self.matrix = []
        # List that keeps trace of the edge associeted to each row
        self.matrix_indices = []
        # To implement lazy computing on other methods
        self.computed = False
    
    @abstractmethod
    def compute(self):
        """ Compute the BitMatrix
        
        Method that is defined in the Concrete istance depending on the
        type of the BitMatrix (QueryBitMatrix, TargetBitMatrix)
        
        :rtype void
        """
        pass

    def _lazy_computing(self):
        """ Method to compute the BitMatrix if it has not been calculated
        
        :rtype void
        """
        not self.is_computed() and self.compute()
        self.computed = True  

    def get_matrix(self):
        """ Retrieve the BitMatrix
        
        It returns only the list of bitmaps, not the edges 
        associeted to all edges
        
        :rtype list of bitmaps
        
        See also
            :func get_matrix_indices()
        """
        self._lazy_computing()
        return self.matrix
    
    def get_graph(self):
        return self.graph
    
    def get_matrix_indices(self):
        """ Retrieve the BitMatrix
        
        :rtype list of bitmaps
        """
        self._lazy_computing()
        return self.matrix_indices
    
    def is_computed(self):
        return self.computed

    def show(self):
        """ Print on terminal the BitMatrix
        
        :rtype void
        """
        self._lazy_computing()
        node_labels = self.graph.get_all_node_labels()
        edge_labels = self.graph.get_all_edge_labels()
        headers = [
            "Tuple",
            f"L_first\n{node_labels}", 
            f"T_in\n{edge_labels}", 
            f"T_out\n{edge_labels}",
            f"L_second\n{node_labels}"
        ]
        data = []
        for i in range(len(self.matrix)):
            row_to_print = [self.matrix_indices[i]] # appending the edge associeted to the bitmap
            row_to_print.extend(self.split_bitmap(i)) # appending the bitmap splitted in the four parts
            data.append(row_to_print)
        table = tabulate(data, headers, tablefmt="fancy_grid")
        print(table)
        
    def split_bitmap(self, row_num):
        """ Return each part of the bitmap associeted to an edge as a string
        
        each bitmap is made up from four parts:
            L_first, T_in, T_out, L_second
        L_first and L_second have num_node_labels elements
        T_in    and T_out have num_edge_labels elements
        
        rtype: list of strings
        """
        self._lazy_computing()
        row = self.matrix[row_num]
        num_node_labels = len(self.graph.get_all_node_labels())
        num_edge_labels = len(self.graph.get_all_edge_labels())
        row_parts = [
            row[:num_node_labels],
            row[num_node_labels:num_node_labels + num_edge_labels],
            row[num_node_labels + num_edge_labels: num_node_labels + 2 * num_edge_labels],
            row[num_node_labels + 2 * num_edge_labels:]
        ]
        return row_parts

class TargetBitMatrix(BitMatrix):
    
    """ Concrete class for describing a BitMatrix for a Target Graph
    
    A Target Graph BitMatrix is a matrix in which the i-th row if a
    bitmap associeted to the edge (a, b) so that the id(a) < id(b).
    We we impose this constraint to keep memory costs low.
    
    :param graph               Graph in which the BitMatrix is computed
    :type  graph               Multigraph
    :param bit_matrix_strategy Datastructure for the rows of the matrix
    :type  bit_matrix_strategy BitMatrixStrategy
    """
    def __init__(self, graph, bit_matrix_strategy):
        super().__init__(graph, bit_matrix_strategy)
    
    def compute(self):
        """ Compute the TargetBitMatrix
        :rtype void
        
        See also:
            :func get()
        """
        super().compute()
        # Extracting edges
        # Using a set to have single occurrence of each edge
        edges = set(self.graph.edges())
        for edge in edges: # edge = (source_id, destination_id)
            # NOTE: there could be this edge e = (a, b)
            # a -> b : id(b) < id(a)
            # checking only id(a) < id(b) and not id(b) < id(a)
            # would bring to a lost of the edge (b, a) in the matrix
            # so we check both conditions
            edge_to_compute = ()
            if(edge[0] < edge[1]):   # condition 1
                edge_to_compute = edge
            elif(edge[1] < edge[0]): # condition 2
                edge_to_compute = (edge[1], edge[0])
            # there could exist an edge (a, b) an (b, a)
            # with the previews if we would consider (b, a) twice
            # so we check if the edges is already been computed
            if(edge_to_compute not in self.matrix_indices): # FIND BETTER METHOD
                # add the bitmap associeted to the edge to the matrix
                self.matrix.append(self.bit_matrix_strategy.compute_row(edge_to_compute))
                # saving the edge associeted to the bitmap
                self.matrix_indices.append(edge_to_compute)

class QueryBitMatrix(BitMatrix):
    
    """ Concrete class for describing a BitMatrix for a Query Graph
    
    A Query Graph BitMatrix is a matrix in which each row associeted to 
    the edge (a, b) with id(a) < id(b) also has the row associeted to
    the edge (b, a). We impose this constraint not to loose any solutions.
    We can "afford" this memory cost because the Query BitMatrix is 
    usually small.
    
    :param graph               Graph in which the BitMatrix is computed
    :type  graph               Multigraph
    :param bit_matrix_strategy Datastructure for the rows of the matrix
    :type  bit_matrix_strategy BitMatrixStrategy
    """
    def __init__(self, graph, bit_matrix_strategy):
        super().__init__(graph, bit_matrix_strategy)
    
    def compute(self):
        """ Compute the QueryBitMatrix
        
        :rtype void
        
        See also:
            :func get()
        """
        super().compute()
        # The comments are the same as the TargetBitMatrix compute
        # method, the small change is at the end.
        edges = set(self.graph.edges())
        for edge in edges: 
            edge_to_compute = ()
            if(edge[0] < edge[1]):
                edge_to_compute = edge
            elif(edge[1] < edge[0]):
                edge_to_compute = (edge[1], edge[0])
            else:
                # This will be executed only if the edge source and destination
                # are the same. To comprend why check how the query graph is
                # constructed
                continue
            if(edge_to_compute not in self.matrix_indices):
                # edge (a, b) bitmap
                self.matrix.append(self.bit_matrix_strategy.compute_row(edge_to_compute))
                self.matrix_indices.append(edge_to_compute)
                # edge (b, a) bitmap
                # do not compute again, we just swap L_first with L_second and T_in with T_out
                i = len(self.matrix) - 1 # index of the bitmap computed before
                row_parts = self.split_bitmap(i) # taking the parts of the bitmap
                self.matrix.append(row_parts[3] + row_parts[2] + row_parts[1] + row_parts[0])
                self.matrix_indices.append((edge_to_compute[1], edge_to_compute[0]))
    
    def _adapt_query_to_target(self, target_graph):
        """ Adding the correct labels to perform the query
        
        The query is constructed on the query graph but it couldn't 
        not have all labels of the target. To handle this situation
        we add a dummy node with all labels that are in the target
        graph. In this way we "force" the query to have all labels
        that have the target. The same thing is done with the edges
        adding loops to the dummy node with all edge labels from the
        target graph.
        
        Example:    Query  labels: x, y
                    Target labels: x, y, z
                    Add a dummy node with labels x, y, z in the query
        """
        self.graph.add_node('dummy', labels=target_graph.get_all_node_labels())
        for label in target_graph.get_all_edge_labels():
            self.graph.add_edge('dummy', 'dummy', type=label)

    def find_candidates(self, target_bitmatrix):
        """ Find the candidate target edges 
        
        This method cycle all query bitmaps (bq) associeted to the edge (a, b)
        and execute an AND operation for each target bitmap (bt) of the edge (x, y).
        If bq & bt == bq then (a, b) and (x, y) could be compatible, thus the 
        tuple candidate ( (a, b), (x, y) ) is added to the list of candidates.
        
        NOTE:   The node a can be mapped to x while b can be mapped to y.
        
        :rtype list of candidates
        """
        self._adapt_query_to_target(target_bitmatrix.get_graph())
        # lazy computing
        self._lazy_computing()
        target_bitmatrix._lazy_computing()
        candidates = []
        bmt = target_bitmatrix.get_matrix()
        bmt_indices = target_bitmatrix.get_matrix_indices()
        # bmq_i is the indices to cycle through the Query  BitMatrix
        # bmt_i is the indices to cycle through the Target BitMatrix
        for bmq_i in range(len(self.matrix)):
            for bmt_i in range(len(bmt)):
                # check if the edge are compatible (read method explanation)
                if(self.matrix[bmq_i] & bmt[bmt_i] == self.matrix[bmq_i]):
                    candidates.append( (self.matrix_indices[bmq_i], bmt_indices[bmt_i]) )
        return candidates