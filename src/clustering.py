import numpy as np
from numpy.typing import NDArray
from src.cluster import Cluster
from src.neuron import Neuron
import src.utils as utils

from typing import Tuple, List, Optional, Any, Dict


def compare(neuron1: Neuron, neuron2: Neuron, weight_epsilon: float = 1e-6, bias_epsilon: float = 1e-6) -> bool:
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


def cluster_algo(neurons: List[Neuron], find_clusters: bool = True) -> Tuple[List[Cluster], List[int]]:
    # Put neurons into clusters
    num_clusters = 0
    clusters: List[Cluster] = []
    for neuron_i, neuron in enumerate(neurons):
        matched = False
        for cluster in clusters:
            if find_clusters and compare(cluster.parent_neuron, neuron):
                cluster.add_neuron(neuron)
                cluster.original_neuron_ids.append(neuron_i)
                matched = True
                break
        if not matched:
            new_cluster = Cluster(cluster_index=num_clusters, parent_neuron=neuron)
            new_cluster.original_neuron_ids.append(neuron_i)
            clusters.append(new_cluster)
            num_clusters += 1

    # Sort neurons
    all_neuron_ind: List[int] = []
    cl_i = -1
    while cl_i < len(clusters) - 1:
        cl_i += 1
        cluster = clusters[cl_i]

        # Get indices associated with functions in cluster
        func_indices = []
        for f_i, f in enumerate(cluster.parent_neuron.functions_per_weight):
            func_indices.extend([f_i for _ in utils.get_all_elements(f.index)])

        ind_list = [
            utils.get_all_elements([f.index for f in n.functions_per_weight])
            for n in neurons
            if n.cluster_index == cluster.cluster_index
        ]
        ind = np.array(ind_list)

        # See if there are any duplicates (potentially offset)
        skip_neuron = get_neurons_to_skip(ind)

        if sum((not s) for s in skip_neuron) > 2:
            lens = [len(np.unique(ind[:, i])) if not skip else 100000 for i, skip in enumerate(skip_neuron)]
            split = np.argmin(lens)
            if lens[split] < 3:
                _, new_ids = np.unique(ind[:, split], return_inverse=True)
                old_neurons = cluster.original_neuron_ids
                num_clusters = len(clusters)
                for i in range(lens[split]):
                    new_neurons = [old_neurons[j] for j, id in enumerate(new_ids) if id == i]
                    new_cluster = Cluster(num_clusters + i - 1, neurons[new_neurons[0]])
                    for j in new_neurons:
                        neurons[j].cluster_index = num_clusters + i - 1
                    new_cluster.original_neuron_ids = new_neurons
                    clusters.append(new_cluster)
                for i in range(cl_i + 1, num_clusters):
                    for j in clusters[i].original_neuron_ids:
                        neurons[j].cluster_index -= 1
                    clusters[i].cluster_index -= 1
                del clusters[cl_i]
                cl_i -= 1
                continue

        # Sort neurons
        all_neuron_ind.extend(cluster.original_neuron_ids)
        # mx_ind = np.max(ind, axis=0)
        # temp_ind = ind.copy().astype(float)
        # for i in range(1,len(mx_ind)):
        #     temp_ind[:,i] = ind[:,i] / np.prod(mx_ind[:i])
        # new_ind = np.argsort(np.sum(temp_ind, axis=1), kind='stable')
        # neuron_ind.extend([cluster.original_neuron_ids[i] for i in new_ind])
        # ind = ind[new_ind]

        val_ind = [i for i in range(ind.shape[1]) if not skip_neuron[i]]
        starts: List[Dict[str, str]] = [{} for _ in range(ind.shape[1])]
        stops: List[Dict[str, str]] = [{} for _ in range(ind.shape[1])]
        cluster.dims = tuple(1 for _ in val_ind)

        order = [0 for _ in val_ind]
        step = len(val_ind)
        while step > 0:
            step -= 1
            outputs = [[{}, {}] for _ in val_ind]
            for o, i in enumerate(val_ind):
                if len(val_ind) > 1:
                    unique_pre, u_indices = np.unique(
                        ind[:, [v for v in val_ind if v != i]], axis=0, return_inverse=True
                    )
                else:
                    unique_pre = np.array([None])
                    u_indices = np.zeros(len(ind))
                v_starts = np.zeros(len(unique_pre), int)
                v_stops = np.zeros(len(unique_pre), int)
                for j in range(len(unique_pre)):
                    v_starts[j] = min(ind[u_indices == j, i])
                    v_stops[j] = max(ind[u_indices == j, i]) + 1

                for s, st in enumerate([v_starts, v_stops]):
                    outputs[o][s] = {"type": "constant", "param": st[0]}

                if any(out is None for out in outputs[o]):
                    linear_combo = utils.find_linear_combo_with_continue(unique_pre, v_starts, v_stops)
                    if linear_combo:
                        if any(linear_combo[0]["params"]):
                            outputs[o] = [{"type": "continue", "param": linear_combo[i]} for i in range(2)]
                        else:
                            outputs[o] = [{"type": "variable", "param": linear_combo[i]} for i in range(2)]

            output_scores = [-i for i in range(len(outputs))]
            for i, o_i in enumerate(outputs):
                for output in o_i:
                    if not output:
                        output_scores[i] += 1000
                    elif output["type"] == "variable":
                        output_scores[i] += 100
                    elif output["type"] == "continue":
                        output_scores[i] += 10
            best_ind = np.argmin(output_scores)
            v = val_ind[best_ind]
            order[step] = v
            starts[step] = outputs[best_ind][0]
            stops[step] = outputs[best_ind][1]
            del val_ind[best_ind]
        val_ind = order

        cluster.layer_table_indices, starts, stops = get_layer_table_indices(starts, stops, cluster)
        cluster.dims = get_dims(cluster.layer_table_indices)
        if all(
            [
                stops[i]["type"] == "constant" and starts[i]["type"] == "constant"
                for i, _ in enumerate(stops)
                if stops[i] and starts[i]
            ]
        ):
            cluster.layer_table_indices = np.array(cluster.layer_table_indices)

        new_starts: List[Dict[str, str]] = [{} for _ in starts]
        new_stops: List[Dict[str, str]] = [{} for _ in starts]
        for i, v in enumerate(val_ind):
            new_starts[v] = starts[i]
            new_stops[v] = stops[i]
        starts = new_starts
        stops = new_stops

        for i in range(len(skip_neuron)):
            if skip_neuron[i]:
                if skip_neuron[i]["type"] == "full":
                    starts[i] = {"type": "skip", "param": "full"}
                    stops[i] = {"type": "skip", "param": "full"}
                elif skip_neuron[i]["type"] == "constant":
                    starts[i] = {"type": None, "param": skip_neuron[i]["value"]}
                    stops[i] = {"type": None, "param": skip_neuron[i]["value"]}
                else:
                    starts[i] = {"type": "skip", "param": skip_neuron[i]["ref"]}
                    stops[i] = {"type": "skip", "param": skip_neuron[i]["ref"]}

        cluster.iterator_inds = [0 for _ in func_indices]
        for i, v in enumerate(val_ind):
            cluster.iterator_inds[v] = i
        for i in range(len(func_indices)):
            if not skip_neuron[i]:
                continue
            if skip_neuron[i]["type"] == "full":
                cluster.iterator_inds[i] = None
            elif skip_neuron[i]["type"] == "constant":
                cluster.iterator_inds[i] = None
            else:
                cluster.iterator_inds[i] = cluster.iterator_inds[skip_neuron[i]["ref"]]

        cluster.starts = starts
        cluster.stops = stops
        cluster.function_ids = func_indices

        if cluster.dims is None:
            cluster.dims = cluster.layer_table_indices.shape
        if len(clusters) > 1:
            cluster.layer_symbol += str(cl_i + 1)

    neurons = [neurons[i] for i in all_neuron_ind]

    return clusters, all_neuron_ind


def get_neurons_to_skip(ind: NDArray[np.int_]) -> List[Dict[str, Any]]:
    skip_neuron: List[Dict[str, Any]] = [{} for _ in range(ind.shape[1])]
    for i in range(ind.shape[1]):
        if skip_neuron[i]:
            continue
        if np.any(ind[:, i] < 0):
            skip_neuron[i] = {"type": "full"}
            continue
        if np.all(ind[:, i] == ind[0, i]):
            skip_neuron[i] = {"type": "constant", "value": ind[0, i]}
            continue
        for j in range(i + 1, ind.shape[1]):
            if skip_neuron[j]:
                continue
            if np.all(ind[:, i] - ind[:, j] == ind[0, i] - ind[0, j]):
                # They are perfect offsets of each other
                skip_neuron[j] = {"type": "offset", "ref": i, "diff": ind[0, j] - ind[0, i]}

    return skip_neuron


def get_layer_table_indices(
    starts: List[Dict[str, Any]], stops: List[Dict[str, Any]], cluster: Cluster
) -> Tuple[List[int], List[Dict[str, Any]], List[Dict[str, Any]]]:
    st_ind = [i for i, st in enumerate(starts) if st and st["type"] != "skip"]
    if len(st_ind) == 0:
        return [0], starts, stops

    range0 = [i for i in range(starts[0]["param"], stops[0]["param"])]
    count = 0
    layer_table_indices, _ = get_partial_indices(1, range0, st_ind, count, starts, stops, [])

    one_side_diagonal = (
        len(st_ind) == 2
        and sum(st[1]["param"][0] == 1 for st in [starts, stops] if st[1]["param"] == "variable") == 1
    )
    one_side_diagonal = one_side_diagonal or (
        len(st_ind) == 2
        and sum(st[1]["param"][0][0] == 1 for st in [starts, stops] if st[1]["param"] == "continue") == 1
    )
    if one_side_diagonal:
        # Let's reflect it
        d = stops[0]["param"] + 1
        cluster.dims = (d, d)
        new_indices = [[None for _ in range(d)] for _ in range(d)]
        upper_matrix = len(layer_table_indices[0]) > len(layer_table_indices[-1])

        sgn = 1
        if cluster.function_type == "mix":
            sgn = -1
        for i, a in enumerate(layer_table_indices):
            for j, b in enumerate(a):
                if upper_matrix:
                    new_indices[i][j + i + 1] = utils.multiply(b, sgn)
                    new_indices[j + i + 1][i] = utils.multiply(b, sgn)
                else:
                    new_indices[i + 1 + j][j] = utils.multiply(b, sgn)
                    new_indices[j][i + 1 + j] = utils.multiply(b, sgn)
        layer_table_indices = new_indices
        starts[1] = {"type": "constant", "param": 0, "symmetric_zero": True}
        stops[1] = {"type": "constant", "param": d, "symmetric_zero": True}
        starts[0] = {"type": "constant", "param": 0}
        stops[0] = {"type": "constant", "param": d}

    return layer_table_indices, starts, stops


def get_partial_indices(
    level: int,
    range_i: List[int],
    st_ind: List[int],
    count: int,
    starts: List[Dict[str, Any]],
    stops: List[Dict[str, Any]],
    upper_ind: List[int],
) -> Tuple[List[Optional[int]], int]:
    if len(range_i) == 0:
        return [], count
    tempi: List[Optional[int]] = [None for _ in range_i]
    for ip, i in enumerate(range_i):
        if len(st_ind) <= level:
            tempi[ip] = count
            count += 1
            continue
        si = st_ind[level]
        upper_ind_i = [u for u in upper_ind]
        upper_ind_i.append(i)

        for s, st in enumerate([starts, stops]):
            if st[si]["type"] == "constant":
                temp = st[si]["param"]
            elif st[si]["type"] == "variable":
                temp = st[si]["param"][-1] + sum(st[si]["param"][i] * u for i, u in enumerate(upper_ind_i))
            else:
                temp = st[si]["param"][0][-1] + sum(st[si]["param"][0][i] * u for i, u in enumerate(upper_ind_i))

            if s == 0:
                start = temp
            else:
                stop = temp

        range_j = [i for i in range(start, stop)]
        tempi[ip], count = get_partial_indices(level + 1, range_j, st_ind, count, starts, stops, upper_ind_i)

    return tempi, count


def get_dims(x: Any) -> Tuple[int, ...]:
    if not isinstance(x, (list, tuple, np.ndarray)):
        return ()
    dims = [len(x)]
    for x_i in x:
        dim = get_dims(x_i)
        for d_i, d in enumerate(dim):
            if d_i + 1 >= len(dims):
                dims.append(d)
            else:
                dims[d_i + 1] = max(d, dims[d_i + 1])
    return tuple(dims)
