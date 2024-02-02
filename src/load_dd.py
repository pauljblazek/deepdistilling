import numpy as np
from numpy.typing import NDArray
import pandas as pd

from src.cluster import Cluster
from src.clustering import cluster_algo
from src.forloop import ForLoop
from src.neuron import Neuron
import src.utils as utils

from typing import Optional, List, Tuple


class MetaLayer:
    input_symbol: str
    layer_symbol: str
    layer_weight_matrix: NDArray[np.float_]
    num_neurons: int
    input_dims: List[Tuple[int, ...]]
    out_dims: List[Tuple[int, ...]]
    neurons: List[Neuron]
    clusters: List[Cluster]
    forloops: List[ForLoop]
    neuron_ind: List[int]

    def __init__(
        self,
        input_symbol: str,
        layer_symbol: str,
        layer_weight_matrix: pd.DataFrame,
        in_dims: List[Tuple[int, ...]],
        ref_ind: Optional[List[List[int]]] = None,
        past_neuron_ind: Optional[int] = None,
        signs: bool = True,
    ) -> None:
        self.input_symbol = input_symbol
        self.layer_symbol = layer_symbol
        self.layer_weight_matrix = layer_weight_matrix.to_numpy()
        if past_neuron_ind is not None:
            self.layer_weight_matrix[:-1, :] = self.layer_weight_matrix[past_neuron_ind, :]
        if not signs:
            self.layer_weight_matrix[-1, :] -= np.sum(self.layer_weight_matrix[:-1, :], axis=0)
            self.layer_weight_matrix[:-1, :] *= 2
        self.num_neurons = layer_weight_matrix.shape[1]
        self.input_dims = in_dims
        self.out_dims = []
        self.neurons = []
        self.clusters = []
        self.forloops = []
        self.neuron_ind = []
        self.signs = True
        if ref_ind:
            self.ref_ind = ref_ind
            mx_ind = 0

            for i in range(len(ref_ind)):
                all_ind: List[int] = utils.get_all_elements(ref_ind[i])
                sgn = 1 - 2 * any(a < 0 for a in all_ind if a is not None)
                ref_ind[i] = utils.add(ref_ind[i], sgn * mx_ind)
                mx_ind += max(abs(a) for a in all_ind if a is not None) + 1
        else:
            self.ref_ind = []
            start_ind = 0
            for dim in in_dims:
                if len(dim) == 1:
                    reference_ind = [j + start_ind for j in range(dim[0])]
                    start_ind += dim[0]
                else:
                    reference_ind = [j + start_ind for j in range(np.prod(dim))]
                    reference_ind = np.array(reference_ind).reshape(dim)
                    start_ind += np.prod(dim)
                self.ref_ind.append(reference_ind)

    def gen_neurons(self, act_fun: bool = True) -> None:
        for index in range(self.num_neurons):
            neuron_vector = self.layer_weight_matrix[:, index]
            neuron_weight_vector = neuron_vector[:-1]
            neuron_bias = neuron_vector[-1]
            neuron = Neuron(
                self.input_symbol,
                self.layer_symbol,
                index,
                neuron_weight_vector,
                neuron_bias,
                self.ref_ind,
                self.input_dims,
                act_fun,
            )
            self.neurons.append(neuron)

    def gen_clusters(self) -> None:
        self.clusters, self.neuron_ind = cluster_algo(self.neurons)
        self.out_dims = []
        for cluster in self.clusters:
            self.out_dims.append(cluster.dims)
            # cluster.print_summary()

    def gen_forloop(self, act_fun: bool, layer_number: int, signs: bool) -> bool:
        for cl_i, cl in enumerate(self.clusters):
            self.forloops.append(ForLoop(cl, act_fun, layer_number == 0, signs=signs))
            self.out_dims[cl_i] = cl.dims
        if any(f.signs for f in self.forloops):
            for f in self.forloops:
                f.signs = True
            return True
        else:
            return False

    def write_code(self, filename: str, act_fun: bool) -> None:
        for cl_i, _ in enumerate(self.clusters):
            if cl_i == 0:
                initialization = "\n"
            else:
                initialization = ""
            mat_num = str(cl_i + 1)
            initialize = True
            if len(self.clusters) < 2:
                mat_num = ""
            if sum(i > 1 for i in self.out_dims[cl_i]) == 0:
                if act_fun:
                    initialization += f"\t{self.layer_symbol}{mat_num} = 0\n"
                else:
                    initialize = False
            elif sum(i > 1 for i in self.out_dims[cl_i]) == 1:
                od = [i for i in self.out_dims[cl_i] if i > 1][0]
                initialization += f"\t{self.layer_symbol}{mat_num} = np.zeros({od})\n"
            else:
                initialization += f"\t{self.layer_symbol}{mat_num} = np.zeros(({self.out_dims[cl_i][0]}, {self.out_dims[cl_i][1]}))\n"
            if not initialize:
                initialization = ""

            self.forloops[cl_i].write_code(filename, initialization)


def load_layer(
    layer_id: int,
    ENN_layers_filename: str,
    input_dims: List[Tuple[int, ...]],
    layer_names: List[str],
    ref_ind: List[int | List[int]],
    past_neuron_ind: List[List[int]],
    act_fun: bool,
    signs: bool,
) -> MetaLayer:
    ENN_layer_df = pd.read_csv(ENN_layers_filename, header=None)
    metalayer = MetaLayer(
        layer_names[0], layer_names[1], ENN_layer_df, input_dims, ref_ind, past_neuron_ind, signs or layer_id == 0
    )
    metalayer.gen_neurons(act_fun)
    metalayer.gen_clusters()
    metalayer.signs = metalayer.gen_forloop(act_fun, layer_id, signs)

    return metalayer


def clear_output_file(filename: str) -> None:
    with open(filename, "w"):
        pass


def write_header(filename: str, function_name: str, input: str, num_inputs: int) -> None:
    with open(filename, "a") as f:
        f.write("import numpy as np\nimport random\n\n")

        def_string = "def " + function_name + "("
        for i in range(num_inputs):
            if i > 0:
                def_string += ", "
            def_string += input
            if num_inputs > 1:
                def_string += str(i + 1)
        f.write(def_string + "):\n")


def write_return(filename: str, layer: MetaLayer, return_probs: str, signs: bool) -> None:
    sym = layer.layer_symbol
    with open(filename, "a") as f:
        if return_probs == "logic":
            if signs:
                f.write(f"\treturn {sym}1 > {sym}2")
            else:
                f.write(f"\treturn {sym}1 and not {sym}2")
        else:
            if len(layer.clusters) > 1:
                string = f"\t{sym} = ["
                for i in range(len(layer.clusters)):
                    if i > 0:
                        string += ", "
                    string += sym + str(i + 1)
                f.write(string + "]")
            if return_probs == "max":
                f.write(f"\r\n\tresults = np.where({sym}==np.max({sym}))[0]")
                f.write("\n\treturn random.choice(results)")
            elif return_probs == "raw":
                f.write(f"\r\n\treturn np.sign({sym})")
            elif return_probs == "logic":
                f.write(f"\r\n\treturn {sym}1 > {sym}2")
            elif return_probs == "probability":
                f.write(f"\r\n\treturn np.exp({sym})/np.sum(np.exp({sym}))")
            else:
                ValueError("Invalid return_style: must be 'max', 'raw', or 'probability'")


def translate_enn(
    ENN_layers_filenames: List[str],
    layer_names: List[str],
    input_dims: List[Tuple[int, ...]],
    problem_name: str,
    return_style: str = "max",
    logic_inputs: bool = False,
) -> None:
    """
    This is the main function for translating the ENN to code
    return_style can be 'max', 'raw', or 'probability'
    """
    out_file = "output_" + problem_name + ".py"
    clear_output_file(out_file)
    write_header(out_file, "ENN_" + problem_name, layer_names[0], len(input_dims))

    layers: List[MetaLayer] = []
    neuron_ind = None
    signs = not logic_inputs
    act_f = [True for _ in ENN_layers_filenames]
    if (return_style != "raw") and (return_style != "logic"):
        act_f[-1] = False

    ref_ind = []
    for f, file in enumerate(ENN_layers_filenames):
        layers.append(load_layer(f, file, input_dims, layer_names[f : f + 2], ref_ind, neuron_ind, act_f[f], signs))
        input_dims = [cl.dims for cl in layers[-1].clusters]
        ref_ind = [cl.layer_table_indices for cl in layers[-1].clusters]
        neuron_ind = layers[-1].neuron_ind
        signs = layers[-1].signs
    for l, layer in enumerate(layers):
        layer.write_code(out_file, act_f[l])
    write_return(out_file, layers[-1], return_style, signs)
