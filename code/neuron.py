import numpy as np

from preprocessing import Preprocessing

from typing import Tuple

class Neuron:
    input_symbol: str
    input_length: Tuple[int, ...]
    layer_symbol: str
    layer_vector_index: int
    weight_vector: np.array
    bias: float
    IDs: 

    def __init__(self, input_symbol: str, layer_symbol: str, layer_vector_index: int, weight_vector: np.array, bias: float, ref_ind, IDs, act_fun=True):
        self.input_symbol = input_symbol
        self.input_length = [len(Preprocessing.get_all_elements(r)) for r in ref_ind]
        self.layer_symbol = layer_symbol
        self.layer_vector_index = layer_vector_index
        self.weight_vector = weight_vector
        self.bias = bias
        self.IDs = IDs
        self.functions_per_weight = []
        self.cluster_index = None
        self.ref_ind = ref_ind

        self.norm_weight_vector, self.norm_bias, self.norm_unique_weights = Preprocessing.process_parameters(
            self.weight_vector, self.bias, act_fun)

        self.__find_functions()


        if len(self.functions_per_weight)<2:
            self.function_type = 'single'
        elif all([f.weights[0]>0 for f in self.functions_per_weight]):
            self.function_type = 'sum'
        elif all([f.weights[0]<0 for f in self.functions_per_weight]):
            self.function_type = 'sum'
        else:
            self.function_type = 'mix'

    def __find_functions(self):
        for d,dims in enumerate(self.IDs):
            symmetric = False
            if len(dims)==2 and dims[0]==dims[1]:
                if all([all([self.ref_ind[d][r][c]==self.ref_ind[d][c][r] for c in range(dims[0])]) for r in range(dims[0])]):
                    if any([i<0 for i in Preprocessing.get_all_elements(self.ref_ind[d]) if i is not None]):
                        symmetric = -1
                    else:
                        symmetric = 1
            if not symmetric:
                norm_unique_weights = self.norm_unique_weights
            else:
                norm_unique_weights = set([abs(w) for w in self.norm_unique_weights])
            for w in norm_unique_weights:
                if np.isnan(w):
                    raise ValueError("Weight is not defined")
                if w == 0:
                    continue
                input_id = str(d+1)
                if len(self.IDs)==1:
                    input_id = ''
                self.__find_function_per_weight(w, dims, self.ref_ind[d], input_id, symmetric)

    def __find_function_per_weight(self, w: float, dims: tuple, ref_ind, input_id, symmetric):

        weight_vector_indices = Preprocessing.where(self.norm_weight_vector, w, True)
        if not symmetric:
            ref_present = Preprocessing.put_indices_in_references(weight_vector_indices, ref_ind)
        else:
            neg_weight_vector_indices = Preprocessing.where(self.norm_weight_vector, -w, True)
            ref_present = Preprocessing.put_indices_in_references(
                weight_vector_indices, ref_ind, neg_weight_vector_indices)

        if not any([abs(i)==1 for i in Preprocessing.get_all_elements(ref_present)]):
            return

        pattern_args = [w, ref_present, symmetric, dims]
        input_functions_check = []
        input_functions_check.append(self.__input_1d_sum(*pattern_args))
        input_functions_check.append(self.__input_matrix_sum(*pattern_args))
        for cutoff in [1,0]:
            input_functions_check.append(self.__input_row_sum(*pattern_args, cutoff))
            input_functions_check.append(self.__input_col_sum(*pattern_args, cutoff))
        input_functions_check.append(self.__input_matrix_minus_row_sum(*pattern_args))
        input_functions_check.append(self.__input_matrix_minus_col_sum(*pattern_args))
        input_functions_check.append(self.__input_off_sum(*pattern_args))
        #input_functions_check.append(self.__input_partial_sum(*pattern_args))
        input_functions_check.append(self.__input_single(*pattern_args))
        if any([i==1 for i in Preprocessing.get_all_elements(ref_present)]):
            input_functions_check.append(self.__input_row_sum(*pattern_args, len_fraction=.75))
            input_functions_check.append(self.__input_col_sum(*pattern_args, len_fraction=.75))
        
        for v in input_functions_check:
            if v is not None:
                if isinstance(v.index, list):
                    if len(v.weights)>1:
                        weights = v.weights
                    else:
                        weights = [v.weights[0] for _ in range(len(v.weights))]
                    for i in range(len(v.index)):
                        self.functions_per_weight.append(Function(v.key, [weights[i]], True, v.index[i], input_id, v.num_indices, missing=v.missing))
                else:
                    v.input_id = input_id
                    self.functions_per_weight.append(v)
        
        #Break apart certain functions
        for i,f in enumerate(self.functions_per_weight):
            if f.key=='off_sum':
                self.functions_per_weight.append(Function("full_sum", f.weights, True, -1, input_id, f.num_indices+1))
                self.functions_per_weight.append(Function("value", [-w for f in f.weights], True, f.index, input_id, 1))
                del self.functions_per_weight[i]

        #Combine similar functions
        for i in range(len(self.functions_per_weight)):
            j = i+1
            while j<len(self.functions_per_weight):
                if self.functions_per_weight[i].compare_nonweights(self.functions_per_weight[j]):
                    self.functions_per_weight[i].weights[0] += self.functions_per_weight[j].weights[0]
                    del self.functions_per_weight[j]
                else:
                    j += 1

        #If empty, do individual functions
        if any([i==1 for i in Preprocessing.get_all_elements(ref_present)]):
            f_pos, f_neg = self.__input_single(*pattern_args, get_all=True)
            if f_pos is not None:
                if len(f_pos.index)>2:
                    self.functions_per_weight.append(Function("part_sum", [w], True, f_pos.index, input_id, num_indices=len(f_pos.index)))
                else:
                    for i in f_pos.index:
                        self.functions_per_weight.append(Function("value", [w], True, i, input_id, num_indices=1))
            if f_neg is not None:
                if len(f_neg.index)>2:
                    self.functions_per_weight.append(Function("part_sum", [-w], True, f_neg.index, input_id, num_indices=len(f_neg.index)))
                else:
                    for i in f_neg.index:
                        self.functions_per_weight.append(Function("value", [-w], True, i, input_id, num_indices=1))

    def __input_single(self, w:float, ref_present, symmetric, dims, get_all=False):
        if not any([i==1 for i in Preprocessing.get_all_elements(ref_present)]):
            return None
        if not any([i>1 for i in dims]):
            if ref_present[0] != 0:
                if get_all:
                    if ref_present[0]>0:
                        ref_present[0] = 0
                        return Function("value", [w], True, [0], num_indices=1), None
                    else:
                        ref_present[0] = 0
                        return None, Function("value", [-w], True, [0], num_indices=1)
                else:
                    ref_present[0] = 0
                    return Function("value", [w], True, 0, num_indices=1)
            return None
        if get_all:
            pos_ind = []
            neg_ind = []
            for r in range(len(ref_present)):
                if isinstance(ref_present[r], (list, tuple, np.ndarray)):
                    for c in range(len(ref_present[r])):
                        if symmetric and c<r:
                            continue
                        if abs(ref_present[r][c])==1:
                            if sum([d>1 for d in dims])==1:
                                index = max(r,c)
                            else:
                                #index = r*dims[1] + c
                                index = (r,c)
                            if ref_present[r][c]>0:
                                pos_ind.append(index)
                            else:
                                neg_ind.append(index)
                else:
                    if ref_present[r]==1:
                        pos_ind.append(r)
                    elif ref_present[r]==-1:
                        neg_ind.append(r)

            return Function("value", [w], True, pos_ind, num_indices=1), Function("value", [-w], True, neg_ind, num_indices=1)
        if sum([i==1 for i in Preprocessing.get_all_elements(ref_present)])==1:
            if sum([d>1 for d in dims])==1:
                for r in range(len(ref_present)):
                    if isinstance(ref_present[r], list):
                        if ref_present[r][0]:
                            index = r
                    elif ref_present[r]:
                        index = r
                    ref_present[r] = 0
            else:
                for r in range(len(ref_present)):
                    for c in range(len(ref_present[r])):
                        if ref_present[r][c]:
                            index = (r,c)
                            if sum([d>1 for d in dims])==1:
                                index = max(index)
                        ref_present[r][c] = 0
            return Function("value", [w], True, index, num_indices=1)
            
        return None
    
    def __input_1d_sum(self, w:float, ref_present, symmetric, dims):
        if not any([i==1 for i in Preprocessing.get_all_elements(ref_present)]):
            return None
        if sum([i>1 for i in dims])>1:
            return None
        if len(ref_present)<2:
            return None
        all_indices = Preprocessing.get_all_elements(ref_present)
        if all(all_indices):
            if len(dims)==2:
                for r in ref_present:
                    for c in range(len(r)):
                        r[c] = 0
            else:
                for r in range(len(ref_present)):
                    ref_present[r] = 0
            return Function("full_sum", [w], True, -1, num_indices=len(all_indices))
        return None

    def __input_matrix_sum(self, w:float, ref_present, symmetric, dims):
        if any([i==1 for i in dims]) or len(dims)==1:
            return None
        if not any([i==1 for i in Preprocessing.get_all_elements(ref_present)]):
            return None
        all_indices = Preprocessing.get_all_elements(ref_present)
        if all(all_indices):
            if len(dims)==2:
                for r in ref_present:
                    for c in range(len(r)):
                        r[c] = 0
            return Function("mat_sum", [w], True, -1, num_indices=len(all_indices))
        return None

    def __input_row_sum(self, w:float, ref_present, symmetric, dims, cutoff=1, len_fraction=1):
        if any([i==1 for i in dims]) or len(dims)==1:
            return None
        if not any([abs(i)==1 for i in Preprocessing.get_all_elements(ref_present)]):
            return None
        row_counts = [sum([r_i!=0 for r_i in r]) for r in ref_present]
        row_lengths = [len(r) for r in ref_present]
        if len_fraction==1:
            if max(row_lengths)>2:
                len_fraction = (max(row_lengths)-1.1)/max(row_lengths)
            full_sum = True
        else:
            full_sum = False
        row_present = [(row_counts[r_i]>=len_fraction*(row_lengths[r_i]-1)) and len(r)>cutoff for r_i,r in enumerate(ref_present)]
        which_rows = [r_i for r_i,r in enumerate(row_present) if r]
        if len(which_rows)>0 and len(which_rows)<dims[1]*.5:
            if not full_sum:
                missing_cols = [[i for i,c in enumerate(row) if c==0] for r_i, row in enumerate(ref_present) if r_i in which_rows]
                if any(col!=missing_cols[0] for col in missing_cols):
                    return None
                missing_cols = missing_cols[0]
            ws = []
            for r in which_rows:
                if ref_present[r][-1]==-1 or ref_present[r][0]==-1:
                    ws.append(-w)
                else:
                    ws.append(w)
                for c in range(len(ref_present[r])):
                    if ref_present[r][c] != 2:
                        if symmetric==1 and ref_present[r][c]==ref_present[c][r]:
                            ref_present[c][r] = 0
                        elif symmetric==-1 and ref_present[r][c]==-ref_present[c][r]:
                            ref_present[c][r] = 0
                        ref_present[r][c] = 0
            if full_sum or len(missing_cols)==0:
                return Function("row_sum", ws, True, which_rows, num_indices=dims[1])
            else:
                return Function("row_sum", ws, True, which_rows, num_indices=dims[1]-len(missing_cols), missing=missing_cols)
        return None

    def __input_col_sum(self, w:float, ref_present, symmetric, dims, cutoff=1, len_fraction=1):
        if any([i==1 for i in dims]) or len(dims)==1:
            return None
        if not any([abs(i)==1 for i in Preprocessing.get_all_elements(ref_present)]):
            return None       
        max_col = max(len(r) for r in ref_present)
        col_lengths = [sum([len(r)>=c for r in ref_present]) for c in range(max_col)]
        if len_fraction==1:
            if max(col_lengths)>2:
                len_fraction = (max(col_lengths)-1.1)/max(col_lengths)
            full_sum = True
        else:
            full_sum = False
        col_counts = [sum([r[c] !=0 for r in ref_present if len(r)>=c]) for c in range(max_col)]
        col_present = [(col_counts[c]>=len_fraction*(col_lengths[c])) and col_lengths[c]>cutoff for c in range(max_col)]
        which_cols = [c_i for c_i, c in enumerate(col_present) if c]
        if len(which_cols)>0 and len(which_cols)<=dims[0]*.5:
            if not full_sum:
                missing_rows = [[r_i for r_i,r in enumerate(ref_present) if r[c]==0] for c in which_cols]
                if any(row!=missing_rows[0] for row in missing_rows):
                    return None
                missing_rows = missing_rows[0]
            ws = []
            for c in which_cols:
                if ref_present[-1][c]==-1 or ref_present[0][c]==-1:
                    ws.append(-w)
                else:
                    ws.append(w)
                for r in range(len(ref_present)):
                    if ref_present[r][c] != 2:
                        if symmetric==1 and ref_present[r][c]==ref_present[c][r]:
                            ref_present[c][r] = 0
                        elif symmetric==-1 and ref_present[r][c]==-ref_present[c][r]:
                            ref_present[c][r] = 0
                        ref_present[r][c] = 0
            if full_sum:
                return Function("col_sum", ws, True, which_cols, num_indices=dims[0])
            else:
                return Function("col_sum", ws, True, which_cols, num_indices=dims[0]-len(missing_rows), missing=missing_rows)
        return None

    def __input_matrix_minus_row_sum(self, w:float, ref_present, symmetric, dims):
        if any([i==1 for i in dims]) or len(dims)==1:
            return None
        if symmetric or not any([i==1 for i in Preprocessing.get_all_elements(ref_present)]):
            return None
        which_row = [r_i for r_i,r in enumerate(ref_present) if (not any(r))]
        if len(which_row)==1 and len(ref_present)>2:
            for r in ref_present:
                for c in range(len(r)):
                    r[c] = 0
            if sum([d>1 for d in dims])==1:
                return Function("off_sum", [w], True, which_row[0], num_indices=np.prod(dims)-1)
            return Function("offrow_sum", [w], True, which_row[0], num_indices=(dims[0]*dims[1]-dims[1]))
        return None
    
    def __input_matrix_minus_col_sum(self, w:float, ref_present, symmetric, dims):
        if any([i==1 for i in dims]) or len(dims)==1:
            return None
        if symmetric or not any([i==1 for i in Preprocessing.get_all_elements(ref_present)]):
            return None
        cols = [not any([r[c] for r in ref_present]) for c in range(dims[1])]
        which_col = [c_i for c_i,c in enumerate(cols) if c]
        if len(which_col)==1 and len(cols)>2:
            for r in ref_present:
                for c in range(len(r)):
                    r[c] = 0
            if sum([d>1 for d in dims])==1:
                return Function("off_sum", [w], True, which_col[0], num_indices=np.prod(dims)-1)
            return Function("offcol_sum", [w], True, which_col[0], num_indices=(dims[0]*dims[1]-dims[0]))
        return None

    def __input_off_sum(self, w:float, ref_present, symmetric, dims):
        if symmetric or not any([i==1 for i in Preprocessing.get_all_elements(ref_present)]):
            return None
        if any([d>1 for d in dims]) != 1:
            return None
        which_el = [r_i for r_i,r in enumerate(ref_present) if (not np.any(r))]
        if len(which_el)==1 and len(ref_present)>2:
            for r in range(len(ref_present)):
                if isinstance(r, (list, tuple, np.ndarray)):
                    for c in range(len(r)):
                        r[c] = 0
                else:
                    ref_present[r] = 0
            return Function("off_sum", [w], True, which_el[0], num_indices=np.prod(dims)-1)
        return None


    

    def __input_partial_sum(self, w:float, rows_present, rows_lens, cols_present, cols_lens, ref_present, dims):
        if not sum([i>1 for i in dims])==1:
            return None
        if not any([i==1 for i in Preprocessing.get_all_elements(ref_present)]):
            return None
        if len(cols_lens)<len(rows_lens):
            indices = [i for i,r in enumerate(rows_lens) if r]
        else:
            indices = [i for i,c in enumerate(cols_lens) if c]
        if len(indices)>1 and len(indices)<sum(rows_lens):
            remaining_indices = indices.copy()
            mx_ind = max(indices)
            sum_indices = []
            for base in [2,3,4,5,6,7,10,12]:
                ex = 1
                increment = base**ex
                while increment < mx_ind:
                    for start in range(increment):
                        temp_ind = np.arange(start, mx_ind, increment)
                        #See if all these indices are in there
                        if len(np.intersect1d(temp_ind, remaining_indices))==len(temp_ind):
                            remaining_indices = np.setdiff1d(remaining_indices, temp_ind)
                            sum_indices.append((start, mx_ind, increment))
                            if len(remaining_indices)==0:
                                break
                    if len(remaining_indices) == 0:
                        break
                    ex += 1
                    increment = base**ex
                if len(remaining_indices) == 0:
                    break
            if len(remaining_indices)==0:
                return Function("partial_sum", [w], sum_indices, num_indices=len(indices))

        return None

    def set_cluster_index(self, cluster_index):
        self.cluster_index = cluster_index


class Function:

    def __init__(self, key, weights, is_present, index=None, input_id=None, num_indices=-1, missing=None):
        self.key = key
        self.weights = weights
        self.is_present = is_present
        self.index = index
        self.input_id = input_id
        self.num_indices = num_indices
        self.missing = missing
    
    def compare_nonweights(self, function):
        if self.key != function.key:
            return False
        if self.is_present != function.is_present:
            return False
        if self.index != function.index:
            return False
        if self.input_id != function.input_id:
            return False
        if self.num_indices != function.num_indices:
            return False
        if self.missing != function.missing:
            return False
        return True
