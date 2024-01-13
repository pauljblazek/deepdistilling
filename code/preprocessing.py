import numpy as np

class Preprocessing:

    @staticmethod
    def process_parameters(weights, bias, rescale=True):
        # Find factor to scale by
        unique_weights = np.array(list(set(np.abs(np.round(weights,6)))))
        unique_weights_nz = unique_weights[unique_weights > 0]
        if rescale:
            factor = np.min(unique_weights_nz)
        else:
            factor = 1
        # Apply factor
        norm_weights = np.round(weights / factor, 3)
        norm_bias = np.round(bias / factor, 3)
            
        
        # Make integers or round to 2 dec places
        
        unique_values = np.concatenate((unique_weights_nz/factor, [norm_bias]))
        if rescale:
            mult = 1
            all_int = 0
            while mult<=10:
                all_int = np.all(np.abs(np.round(unique_values*mult, 0) - unique_values*mult)<.01)
                if all_int:
                    break
                mult += 1
            if all_int:
                norm_bias *= mult
                norm_weights *= mult
                norm_bias = int(round(norm_bias,0))
                norm_weights = np.round(norm_weights,0).astype(int)
            else:
                norm_weights = np.around(norm_weights, 2)
                norm_bias = np.around(norm_bias, 2)
        
        # Compute unique weights (+/-)
        norm_unique_weights = np.array(list(set(norm_weights)))

        return norm_weights, norm_bias, norm_unique_weights

    @staticmethod
    # unrolls via columns (primary vector --> table conversion function, used in program)
    def vector_to_table(vector_indices, dims, inputs=1):
        table_indices = []
        for vector_index in vector_indices:
            table_index = [-1, -1]
            table_index[1] = vector_index % dims[0]
            table_index[0] = vector_index // dims[0]
            table_indices.append(table_index)
        return table_indices

    @staticmethod
    # unrolls via rows
    def vector_to_table_alt(vector_indices, dims, inputs=1):
        table_indices = []
        for vector_index in vector_indices:
            table_index = [-1, -1]
            table_index[0] = vector_index // dims[0]
            table_index[1] = vector_index % dims[0]
            table_indices.append(table_index)
        return table_indices

    @staticmethod
    def table_to_vector(table_indices, dims, inputs=1):
        vector_indices = []
        for table_index in table_indices:
            vector_index = table_index[0] * dims[1] + table_index[1]
            vector_indices.append(vector_index)
        return vector_indices
    
    @staticmethod
    def get_indices_from_reference(indices, ref):
        ref_ind = [[] for i in range(len(indices))]
        for i in range(len(ref)):
            for j in range(len(ref[i])):
                try:
                    temp_ref = indices.index(ref[i][j])
                    ref_ind[temp_ref] = [i,j]
                except ValueError:
                    continue
        return ref_ind
            
    @staticmethod
    def put_indices_in_references(indices, ref, neg_indices=None):
        flip_rc_sign = 1
        if any([i<0 for i in Preprocessing.get_all_elements(ref) if i is not None]):
            flip_rc_sign = -1
        if neg_indices is None:
            if any([isinstance(r, (list, tuple, np.ndarray)) for r in ref]):
                return [[(c in indices)*flip_rc_sign + 2*(c is None) for c in r] for r in ref]
            else:
                return [(r in indices)*flip_rc_sign + 2*(r is None) for r in ref]
        else:
            result = []
            for r in range(len(ref)):
                result_r = []
                for c in range(len(ref[r])):
                    if ref[r][c] is None:
                        result_r.append(2)
                        continue
                    sgn = 1
                    if flip_rc_sign==-1:
                        sgn = 1-2*(r>c)
                    if abs(ref[r][c]) in indices:
                        result_r.append(sgn)
                    elif abs(ref[r][c]) in neg_indices:
                        result_r.append(-sgn)
                    else:
                        result_r.append(0)
                result.append(result_r)
            return result

    @staticmethod
    def where(vector, w, bool):
        indices = []
        n = vector.shape[0]
        for i in range(n):
            if (vector[i] == w) == bool:
                indices.append(i)
        return indices
    
    @staticmethod
    def get_all_elements(x):
        if not isinstance(x, (list, tuple, np.ndarray)):
            return [x]
        new_x = []
        for i in x:
            out_x = Preprocessing.get_all_elements(i)
            new_x.extend(out_x)
        return new_x
    
    @staticmethod
    def multiply(x, a):
        if a==1:
            return x
        if not isinstance(x, (list, tuple, np.ndarray)):
            return x*a
        x = [Preprocessing.multiply(i, a) for i in x]
        return x
    
    @staticmethod
    def add(x, a):
        if a==0 or x is None:
            return x
        if not isinstance(x, (list, tuple, np.ndarray)):
            return x+a
        x = [Preprocessing.add(i, a) for i in x]
        return x

    @staticmethod
    def zeros(ref):
        if isinstance(ref, np.ndarray):
            return np.zeros(ref.shape)
        if not isinstance(ref, (list, tuple, np.ndarray)):
            return 0
        zs = []
        for i in ref:
            out_zs = Preprocessing.zeros(i)
            zs.append(out_zs)
        return zs
    
    @staticmethod
    def find_linear_combo(x, y):
        possible_values = [0,1,-1]
        n = len(x[0])
        curr_pos = np.zeros(n).astype(int)
        while True:
            y_pred = np.zeros(y.shape)
            for i in range(n):
                y_pred += x[:,i]*possible_values[curr_pos[i]]
            b = int(y[0]-y_pred[0])
            y_pred += b
            if np.all(y==y_pred):
                factors = [int(possible_values[p]) for p in curr_pos]
                factors.append(b)
                return factors
            
            keep_going = False
            for i in range(n):
                if curr_pos[i]==len(possible_values)-1:
                    curr_pos[i] = 0
                else:
                    curr_pos[i] += 1
                    keep_going = True
                    break
            if not keep_going:
                return False
    
    @staticmethod
    def find_linear_combo_with_continue(x, y1, y2):
        n = len(x[0])
        continues = [[(i % 2**(j+1))>=(2**j) for j in range(n)] for i in range(2**n)]
        
        for cont in continues:
            lins = [None, None]
            for y_i,y in enumerate([y1, y2]):
                temp_x = x
                temp_y = y
                found = not any(cont)
                for i,c in enumerate(cont):
                    if c:
                        ind = np.where(temp_x[:,i]==temp_y)[0]
                        if len(ind)>0:
                            found = True
                            temp_x = np.delete(temp_x, ind, axis=0)
                            temp_y = np.delete(temp_y, ind)
                if found:
                    lin = Preprocessing.find_linear_combo(temp_x, temp_y)
                    if lin:
                        lins[y_i] = lin
                    else:
                        break
            if all(l is not None for l in lins):
                return [(lins[0], cont), (lins[1], cont)]

        return False

            