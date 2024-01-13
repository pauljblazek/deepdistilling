import numpy as np
from cluster import Cluster
from preprocessing import Preprocessing

def compare(neuron1, neuron2, weight_epsilon=1e-6, bias_epsilon=1e-6):
    if abs(neuron1.bias - neuron2.bias) > bias_epsilon:
        return False
    if len(neuron1.functions_per_weight) != len(neuron2.functions_per_weight):
        return False
    for func1, func2 in zip(neuron1.functions_per_weight, neuron2.functions_per_weight):
        if func1.key != func2.key:
            return False
        if len(func1.weights) != len(func2.weights):
            return False
        if func1.input_id != func2.input_id:
            return False
        diffs = [abs(w1 - w2) for w1, w2 in zip(func1.weights, func2.weights)]
        for delta in diffs:
            if delta > weight_epsilon:
                return False
    return True
    

def cluster_algo(neurons, find_clusters=True):
    init_cluster = Cluster(0, neurons[0])
    clusters = [init_cluster]
    cluster_index = 1
    init_cluster.original_neuron_ids.append(0)
    
    for it in range(1, len(neurons)):
        matched = False
        for cl in clusters:
            if find_clusters and compare(cl.parent_neuron, neurons[it]):
                cl.add_neuron(neurons[it])
                cl.original_neuron_ids.append(it)
                matched = True
                break
        if not matched:
            new_cluster = Cluster(cluster_index, neurons[it])
            new_cluster.original_neuron_ids.append(it)
            clusters.append(new_cluster)
            cluster_index+=1
    for it in range(len(neurons)):
        if neurons[it].cluster_index is None:
            raise ValueError("Algorithm failed, nueuron {} not assigned a cluster".format(it))

    #Sort neurons
    neuron_ind = []
    cl_i = -1
    while cl_i<len(clusters)-1:
        cl_i += 1
        cl = clusters[cl_i]
        f_ids = []
        for f_i,f in enumerate(cl.parent_neuron.functions_per_weight):
            f_ids.extend([f_i for _ in Preprocessing.get_all_elements(f.index)])
        ind = [Preprocessing.get_all_elements([f.index for f in n.functions_per_weight]) for n in neurons if n.cluster_index==cl.cluster_index]
        ind = np.array(ind)      

        #See if there are any duplicates (potentially offset)
        skip = [None for _ in range(ind.shape[1])]
        for i in range(ind.shape[1]):
            if skip[i] is not None:
                continue
            if np.any(ind[:,i]<0):
                skip[i] = 'full'
                continue
            if np.all(ind[:,i]==ind[0,i]):
                skip[i] = ('constant', ind[0,i])
                continue
            for j in range(i+1,ind.shape[1]):
                if skip[j] is not None:
                    continue
                if np.all(ind[:,i]-ind[:,j]==ind[0,i]-ind[0,j]):
                    #They are perfect offsets of each other
                    skip[j] = (i, ind[0,j]-ind[0,i])

        if sum([s is None for s in skip])>2:
            lens = [len(np.unique(ind[:,i])) for i in range(ind.shape[1])]
            for s_i,s in enumerate(skip):
                if s is not None:
                    lens[s_i] = np.inf
            split = np.argmin(lens)
            if lens[split]<3:
                _, new_ids = np.unique(ind[:,split], return_inverse=True)
                old_neurons = cl.original_neuron_ids
                num_clusters = len(clusters)
                for i in range(lens[split]):
                    new_neurons = [old_neurons[j] for j,id in enumerate(new_ids) if id==i]
                    new_cluster = Cluster(num_clusters+i-1, neurons[new_neurons[0]])
                    for j in new_neurons:
                        neurons[j].cluster_index = num_clusters+i-1
                    new_cluster.original_neuron_ids = new_neurons
                    clusters.append(new_cluster)
                for i in range(cl_i+1, num_clusters):
                    for j in clusters[i].original_neuron_ids:
                        neurons[j].cluster_index -= 1
                    clusters[i].cluster_index -= 1
                del clusters[cl_i]
                for c in clusters:
                    for o in c.original_neuron_ids:
                        if neurons[o].cluster_index != c.cluster_index:
                            debug=1
                cl_i -= 1
                continue

        #Sort neurons
        neuron_ind.extend([i for i in cl.original_neuron_ids])
        """
        mx_ind = np.max(ind, axis=0)
        temp_ind = ind.copy().astype(float)
        for i in range(1,len(mx_ind)):
            temp_ind[:,i] = ind[:,i] / np.prod(mx_ind[:i])
        new_ind = np.argsort(np.sum(temp_ind, axis=1), kind='stable')
        neuron_ind.extend([cl.original_neuron_ids[i] for i in new_ind])
        ind = ind[new_ind]
        """

        val_ind = [i for i in range(ind.shape[1]) if skip[i] is None]
        starts = [None for _ in range(ind.shape[1])]
        stops = [None for _ in range(ind.shape[1])]
        cl.dims = [1 for _ in range(len(val_ind))]

        order = [0 for _ in val_ind]
        step = len(val_ind)
        while step>0:
            step -= 1
            outputs = [[None, None] for _ in val_ind]
            for o,i in enumerate(val_ind):
                if len(val_ind)>1:
                    unique_pre, u_indices = np.unique(ind[:,[v for v in val_ind if v!=i]], axis=0, return_inverse=True)
                else:
                    unique_pre = ['']
                    u_indices = np.zeros(len(ind))
                v_starts = np.zeros(len(unique_pre), int)
                v_stops = np.zeros(len(unique_pre), int)
                for j in range(len(unique_pre)):
                    v_starts[j] = min(ind[u_indices==j,i])
                    v_stops[j] = max(ind[u_indices==j,i])+1
                
                for s,st in enumerate([v_starts, v_stops]):
                    if np.all(st==st[0]):
                        outputs[o][s] = ('i', st[0])
                
                if any(out is None for out in outputs[o]):
                    lin = Preprocessing.find_linear_combo_with_continue(unique_pre, v_starts, v_stops)
                    if lin:
                        if any(lin[0][0]):
                            outputs[o][0] = ('c', lin[0])
                            outputs[o][1] = ('c', lin[1])
                        else:
                            outputs[o][0] = ('v', lin[0][0])
                            outputs[o][1] = ('v', lin[1][0])
            
            output_scores = [-i for i in range(len(outputs))]
            for i, o_i in enumerate(outputs):
                for o in o_i:
                    if o is None:
                        output_scores[i] += 1000
                    elif o[0] == 'v':
                        output_scores[i] += 100
                    elif o[0] == 'c':
                        output_scores[i] += 10
            best_ind = np.argmin(output_scores)
            v = val_ind[best_ind]
            order[step] = v
            starts[step] = outputs[best_ind][0]
            stops[step] = outputs[best_ind][1]
            del val_ind[best_ind]
        val_ind = order

        cl.layer_table_indices, starts, stops = get_layer_table_indices(starts, stops, cl)
        cl.dims = get_dims(cl.layer_table_indices)
        if all([stops[i][0]=='i' and starts[i][0]=='i' for i,_ in enumerate(stops) if stops[i] is not None and starts[i] is not None]):
            cl.layer_table_indices = np.array(cl.layer_table_indices)
        
        new_starts = [None for _ in starts]
        new_stops = [None for _ in starts]
        for i,v in enumerate(val_ind):
            new_starts[v] = starts[i]
            new_stops[v] = stops[i]
        starts = new_starts
        stops = new_stops

        for i in range(len(skip)):
            if skip[i] is not None:
                if skip[i]=='full':
                    starts[i] = ('s', 'full')
                    stops[i] = ('s', 'full')
                elif skip[i][0]=='constant':
                    starts[i] = (None, skip[i][1])
                    stops[i] = (None, skip[i][1])
                else:
                    starts[i] = ('s',skip[i][0])
                    stops[i] = ('s', skip[i][0])        

        cl.iterator_inds = [0 for _ in f_ids]
        for i,v in enumerate(val_ind):
            cl.iterator_inds[v] = i
        for i in range(len(f_ids)):
            if skip[i]=='full':
                cl.iterator_inds[i] = None
            elif skip[i] is not None:
                if skip[i][0]=='constant':
                    cl.iterator_inds[i] = None
                else:
                    cl.iterator_inds[i] = cl.iterator_inds[skip[i][0]]

        cl.starts = starts
        cl.stops = stops
        cl.function_ids = f_ids
            
        if cl.dims is None:
            cl.dims = cl.layer_table_indices.shape
        if len(clusters)>1:
            cl.layer_symbol += str(cl_i+1)

    neurons = [neurons[i] for i in neuron_ind]
        


    return clusters, neuron_ind

def get_layer_table_indices(starts, stops, cluster):
    st_ind = [i for i,st in enumerate(starts) if st is not None and st[0]!='s']
    if len(st_ind)==0:
        return [0], starts, stops

    range0 = range(starts[0][1], stops[0][1])
    count = 0
    layer_table_indices,_ = get_partial_indices(1, range0, st_ind, count, starts, stops, [])

    one_side_diagonal = len(st_ind) == 2 and sum([st[1][1][0]==1 for st in [starts, stops] if st[1][0]=='v'])==1
    one_side_diagonal = one_side_diagonal or (len(st_ind) ==2 and sum([st[1][1][0][0]==1 for st in [starts, stops] if st[1][0]=='c'])==1)
    if one_side_diagonal:
        #Let's reflect it
        d = stops[0][1]+1
        cluster.dims = (d,d)
        new_indices = [[None for _ in range(d)] for _ in range(d)]
        upper_matrix = len(layer_table_indices[0])>len(layer_table_indices[-1])
        
        sgn = 1
        if cluster.function_type == 'mix':
            sgn = -1
        for i,a in enumerate(layer_table_indices):
            for j,b in enumerate(a):
                if upper_matrix:
                    new_indices[i][j+i+1] = Preprocessing.multiply(b, sgn)
                    new_indices[j+i+1][i] = Preprocessing.multiply(b, sgn)
                else:
                    new_indices[i+1+j][j] = Preprocessing.multiply(b, sgn)
                    new_indices[j][i+1+j] = Preprocessing.multiply(b, sgn)
        layer_table_indices = new_indices
        starts[1] = ('i', 0, 'symmetric_zero')
        stops[1] = ('i', d, 'symmetric_zero')
        starts[0] = ('i', 0)
        stops[0] = ('i', d)

    return layer_table_indices, starts, stops

def get_partial_indices(level, range_i, st_ind, count, starts, stops, upper_ind):
    if len(range_i)==0:
        return None, count
    tempi = [None for _ in range_i]
    for ip, i in enumerate(range_i):
        if len(st_ind)<=level:
            tempi[ip] = count
            count += 1
            continue
        si = st_ind[level]
        upper_ind_i = [u for u in upper_ind]
        upper_ind_i.append(i)        

        for s, st in enumerate([starts, stops]):
            if st[si][0] == 'i':
                temp = st[si][1]
            elif st[si][0] == 'v':
                temp = st[si][1][-1] + sum([st[si][1][i]*u for i,u in enumerate(upper_ind_i)])
            else:
                temp = st[si][1][0][-1] + sum([st[si][1][0][i]*u for i,u in enumerate(upper_ind_i)])
            
            if s==0:
                start = temp
            else:
                stop = temp
        
        range_j = range(start, stop)
        tempi[ip], count = get_partial_indices(level+1, range_j, st_ind, count, starts, stops, upper_ind_i)
    
    return tempi, count

def get_dims(x):
    if not isinstance(x, (list, tuple, np.ndarray)):
        return []
    dims = [len(x)]
    for x_i in x:
        dim = get_dims(x_i)
        for d_i,d in enumerate(dim):
            if d_i+1>=len(dims):
                dims.append(d)
            else:
                dims[d_i+1] = max(d, dims[d_i+1])
    return dims

