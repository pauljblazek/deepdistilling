import numpy as np
import pandas as pd

from cluster import Cluster
from clustering import cluster_algo
from forloop import ForLoop
from neuron import Neuron
from preprocessing import Preprocessing

class MetaLayer:

    def __init__(self, input_symbol: chr, layer_symbol: chr, layer_weight_matrix, ref_ind=None, input_dims=None, past_neuron_ind=None, signs=True):
        self.input_symbol = input_symbol
        self.layer_symbol = layer_symbol
        self.layer_weight_matrix = layer_weight_matrix.to_numpy()
        if past_neuron_ind is not None:
            self.layer_weight_matrix[:-1,:] = self.layer_weight_matrix[past_neuron_ind,:]
        if not signs:
            self.layer_weight_matrix[-1,:] -= np.sum(self.layer_weight_matrix[:-1,:], axis=0)
            self.layer_weight_matrix[:-1,:] *= 2
        self.num_neurons = layer_weight_matrix.shape[1]
        self.IDs = input_dims
        self.ODs = []
        self.neurons = []
        self.clusters = []
        self.forloops = []
        self.signs = True
        if ref_ind is None:
            self.ref_ind = []
            start_ind = 0
            for i in input_dims:
                if len(i)==1:
                    reference_ind = [j+start_ind for j in range(i[0])]
                    start_ind += i[0]
                else:
                    reference_ind = [j+start_ind for j in range(np.prod(i))]
                    reference_ind = np.array(reference_ind).reshape(i)
                    start_ind += np.prod(i)
                self.ref_ind.append(reference_ind)
        else:
            self.ref_ind = ref_ind
            mx_ind = 0

            for i in range(len(ref_ind)):
                all_ind = Preprocessing.get_all_elements(ref_ind[i])
                sgn = 1-2*any(a<0 for a in all_ind if a is not None)
                ref_ind[i] = Preprocessing.add(ref_ind[i], sgn*mx_ind)
                mx_ind += max([abs(a) for a in all_ind if a is not None]) + 1
            
        

    def gen_neurons(self, act_fun=True):
        for index in range(self.num_neurons):
            neuron_vector = self.layer_weight_matrix[:, index]
            neuron_weight_vector = neuron_vector[:-1]
            neuron_bias = neuron_vector[-1]
            neuron = Neuron(self.input_symbol, self.layer_symbol, index,
                            neuron_weight_vector, neuron_bias, self.ref_ind, self.IDs, act_fun)
            self.neurons.append(neuron)

    def gen_clusters(self):
        self.clusters, self.neuron_ind = cluster_algo(self.neurons)
        self.ODs = []
        for cl in self.clusters:
            self.ODs.append(cl.dims)
            cl.get_summary()

    def gen_forloop(self, act_fun, layer_number, signs):
        for cl_i,cl in enumerate(self.clusters):
            self.forloops.append(ForLoop(cl, act_fun, layer_number==0, signs=signs))
            self.ODs[cl_i] = cl.dims
        if np.any([f.signs for f in self.forloops]):
            for f in self.forloops:
                f.signs = True
            return True
        else:
            return False

    def write_code(self, filename: str, act_fun: bool):
        
        for cl_i,_ in enumerate(self.clusters):
            if cl_i==0:
                initialization = '\n'
            else:
                initialization = ''
            mat_num = str(cl_i+1)
            initialize = True
            if len(self.clusters)<2:
                mat_num = ''
            if sum([i>1 for i in self.ODs[cl_i]])==0:
                if act_fun:
                    initialization += "\t{}{} = 0\n".format(
                        self.layer_symbol, mat_num)
                else:
                    initialize = False
            elif sum([i>1 for i in self.ODs[cl_i]])==1:
                od = [i for i in self.ODs[cl_i] if i>1][0]
                initialization += "\t{}{} = np.zeros({})\n".format(
                    self.layer_symbol, mat_num, od)
            else:
                initialization += "\t{}{} = np.zeros(({}, {}))\n".format(
                    self.layer_symbol, mat_num, self.ODs[cl_i][0], self.ODs[cl_i][1])
            if not initialize:
                initialization = ''

            self.forloops[cl_i].write_code(filename, initialization)

def load_layer(layer, ENN_layers_filename, input_dims, layer_names, ref_ind, past_neuron_ind, act_fun, signs):
    #print("Layer",layer+1,"---------------------")
    ENN_layer_df = pd.read_csv(ENN_layers_filename, header=None)
    Layer = MetaLayer(layer_names[0], layer_names[1], ENN_layer_df, ref_ind, input_dims, past_neuron_ind, signs or layer==0)
    Layer.gen_neurons(act_fun)
    Layer.gen_clusters()
    Layer.signs = Layer.gen_forloop(act_fun, layer, signs)
    
    return Layer

def clear_output_file(filename):
    with open(filename, "w"):
        pass

def write_header(filename, function_name, input, num_inputs):
    with open(filename, "a") as f:
        f.write("import numpy as np\nimport random\n\n")
        
        def_string = "def " + function_name + "("
        for i in range(num_inputs):
            if i>0:
                def_string += ", "
            def_string += input
            if num_inputs>1:
                def_string += str(i+1)
        f.write(def_string+"):\n")

def write_return(filename, layer: MetaLayer, return_probs, signs):
    sym = layer.layer_symbol
    with open(filename, "a") as f:        
        if return_probs=='logic':
            if signs:
                f.write('\treturn {}1 > {}2'.format(sym, sym))
            else:
                f.write('\treturn {}1 and not {}2'.format(sym, sym))
        else:
            if len(layer.clusters)>1:
                string = "\t{} = [".format(sym)
                for i in range(len(layer.clusters)):
                    if i>0:
                        string += ", "
                    string += sym + str(i+1)                
                f.write(string + "]")
            if return_probs=='max':
                f.write('\r\n\tresults = np.where({}==np.max({}))[0]'.format(sym, sym))
                f.write('\n\treturn random.choice(results)'.format(sym))
            elif return_probs=='raw':
                f.write('\r\n\treturn np.sign({})'.format(sym))
            elif return_probs=='logic':
                f.write('\r\n\treturn {}1 > {}2'.format(sym, sym))
            elif return_probs=='probability':
                f.write('\r\n\treturn np.exp({})/np.sum(np.exp({}))'.format(sym, sym))
            else:
                ValueError('Invalid return_style: must be \'max\', \'raw\', or \'probability\'')

def translate_enn(ENN_layers_filenames, layer_names, input_dims, problem_name, return_style='max', ref_ind=None, logic_inputs=False):
    """
    This is the main function for translating the ENN to code
    return_style can be 'max', 'raw', or 'probability'
    """
    out_file = "output_" + problem_name + ".py"
    clear_output_file(out_file)
    write_header(out_file, "ENN_"+problem_name, layer_names[0], len(input_dims))

    layers = []
    neuron_ind = None
    signs = not logic_inputs
    act_f = [True for _ in enumerate(ENN_layers_filenames)]
    if (return_style != 'raw') and (return_style != 'logic'):
        act_f[-1] = False

    for f,file in enumerate(ENN_layers_filenames):
        layers.append(load_layer(f, file, input_dims, layer_names[f:f+2], ref_ind, neuron_ind, act_f[f], signs))
        input_dims = [cl.dims for cl in layers[-1].clusters]
        ref_ind = [cl.layer_table_indices for cl in layers[-1].clusters]
        neuron_ind = layers[-1].neuron_ind
        signs = layers[-1].signs
    for l,layer in enumerate(layers):
        layer.write_code(out_file, act_f[l])
    write_return(out_file, layers[-1], return_style, signs)
