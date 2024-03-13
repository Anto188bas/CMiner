# pylint: disable=missing-class-docstring
from abc import ABC, abstractmethod
from bitarray import bitarray
from bitstring import BitArray

class BitMatrixStrategy(ABC):
    
    def __init__(self):
        self.graph = None
    
    def set_graph(self, graph):
        self.graph = graph
        
    @abstractmethod
    def compute_row(self):
        """ Compute the BitMatrix row
        
        Method that is defined in the Concrete istance depending on the
        type of the bitmap used
        
        :rtype bitmap
        """
        pass
    
    def _get_row_string(self, edge):
        """ Compute the bitmap with a string format
        
        :rtype str
        """
        source      = edge[0]
        destination = edge[1]
        # compute the bitmaps strings for each part of the row
        L_first  = self._compute_node_bitmap(source)
        L_second = self._compute_node_bitmap(destination)
        tao_in   = self._compute_edge_bitmap((destination, source))
        tao_out  = self._compute_edge_bitmap(edge)
        # concatenate the strings
        return L_first + tao_in  + tao_out + L_second
    
    def _compute_node_bitmap(self, node):
        """
        Given a node this method comput the bitmap of the labels of that node
        
        Example:    all node labels in the graph [x, y, z]
                    node label [x, y]
                    bitmap string [110]
        """
        return ''.join('1' if label in self.graph.get_node_labels(node) else '0' for label in self.graph.get_all_node_labels())

    
    def _compute_edge_bitmap(self, edge):
        """
        Given an edge this method comput the bitmap of the label 
        of that edge
        
        Example:    all edge labels in the graph [a, b, c, d]
                    edge label [a, c]
                    bitmap string [1010]
                    
        NOTE:       the edges are directed so edge[0] is the 
                    source and edge[0] the destination
        """
        return ''.join('1' if label in self.graph.get_edge_labels(edge[0], edge[1]) else '0' for label in self.graph.get_all_edge_labels())




class BitMatrixStrategy1(BitMatrixStrategy):
    
    def __init__(self):
        super().__init__()
        
    """ Convert the string bitmap in a bitmap
    
    It uses bitarray library
    
    :rtype bitmap
    """
    def compute_row(self, edge):
        return bitarray(super()._get_row_string(edge))        
    
    
class BitMatrixStrategy2(BitMatrixStrategy):
    
    def __init__(self):
        super().__init__()
        
    """ Convert the string bitmap in a bitmap
    
    It uses bitarray library
    
    :rtype bitmap
    """
    def compute_row(self, edge):
        return BitArray(bin=super()._get_row_string(edge)) 