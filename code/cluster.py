from neuron import Neuron

from typing import List, Tuple, Optional


class Cluster:
    cluster_index: int
    parent_neuron: Neuron
    bias: float
    key_weight_paris: List[Tuple]
    layer_table_indices: List
    dims: Tuple[int, ...]
    layer_symbol: str
    start_ind: int
    original_neuron_ids: List
    function_type: int
    starts: Optional[List]
    stops: Optional[List]
    function_ids: Optional[List[int]]
    iterator_inds: Optional[List[int]]


    def __init__(self, cluster_index: int, parent_neuron: Neuron) -> None:
        self.cluster_index = cluster_index
        self.parent_neuron = parent_neuron
        self.bias = self.parent_neuron.norm_bias
        self.key_weight_pairs = []
        self.layer_table_indices = []
        self.dims = ()
        self.layer_symbol = parent_neuron.layer_symbol
        self.start_ind = 0
        self.original_neuron_ids = []
        self.function_type = parent_neuron.function_type
        self.starts = None
        self.stops = None
        self.function_ids = None
        self.iterator_inds = None

        self.__set_key_weight_pairs()
        parent_neuron.set_cluster_index(cluster_index)

    def add_neuron(self, neuron: Neuron) -> None:
        neuron.set_cluster_index(self.cluster_index)

    def __set_key_weight_pairs(self) -> None:
        for func in self.parent_neuron.functions_per_weight:
            self.key_weight_pairs.append((func.key, func.weights, func.index, func.input_id, func.missing))

    def print_summary(self) -> None:
        print("Cluster index {} summary----".format(self.cluster_index))
        print("key_weight_pairs: {}".format(self.key_weight_pairs))
        print("cluster bias: {}".format(self.bias))
        print("layer_table_indices: {}".format(self.layer_table_indices))
        print("----------------------------")
        print()
