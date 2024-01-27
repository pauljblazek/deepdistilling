from src.cluster import Cluster
import src.utils as utils
import math
import numpy as np

from typing import Optional, Tuple, List, Any, Dict


def floor(x: float | int) -> int:
    return math.floor(x) if x > 0 else math.ceil(x)


def ceil(x: float | int) -> int:
    return math.ceil(x) if x > 0 else math.floor(x)


class ForLoop:
    place_holder: str = "INSIDE_LOOP"
    cluster: Cluster
    loop_pattern: Optional[str]
    fixed_index: Optional[int]
    num_iterators: int
    assignment_string: str
    function_string: str
    loop_string: str
    flobal_found: bool
    act: bool
    symbolic_inputs: bool
    initialize: bool
    signs: bool
    previous_signs: bool
    iterator_symbols: List[int]

    def __init__(self, cluster: Cluster, act_fun: bool, first_layer: bool = False, signs: bool = True) -> None:
        self.cluster = cluster
        self.loop_pattern = None
        self.fixed_index = None
        self.num_iterators = 0
        self.iterator_symbols = []
        self.assignment_string = ""
        self.function_string = ""
        self.loop_string = ""
        self.global_found = False
        self.act = act_fun
        self.symbolic_inputs = not first_layer
        self.initialize = True
        self.signs = True
        self.previous_signs = signs

        self.find_cluster_structure()
        self.set_function_string()

    def find_cluster_structure(self) -> None:
        def stand_alone(cluster: Cluster) -> bool:
            all_indices = utils.get_all_elements(cluster.layer_table_indices)
            if len(all_indices) == 1:
                return True
            return False

        def one_dim_sum(cluster: Cluster) -> bool:
            if len(cluster.dims) > 1:
                if sum(i != 1 for i in cluster.dims) > 1:
                    return False
            all_indices = utils.get_all_elements(cluster.layer_table_indices)
            if len(set(all_indices)) == len(all_indices):
                return True
            return False

        def upper_matrix_sum(cluster: Cluster) -> bool:
            if any(i == 1 for i in cluster.dims):
                return False
            for i in range(cluster.dims[0]):
                if len(cluster.layer_table_indices[i]) != cluster.dims[1] - i:
                    return False
            sz = len(cluster.layer_table_indices) + 1
            new_indices: List[List[Optional[int]]] = [[None for _ in range(sz)] for _ in range(sz)]
            sgn = 1
            if cluster.function_type == "mix":
                sgn = -1
            for r in range(sz - 1):
                for c in range(len(cluster.layer_table_indices[r])):
                    col = c + sz - 1 - len(cluster.layer_table_indices[r])
                    new_indices[r][col + 1] = sgn * cluster.layer_table_indices[r][c]
                    new_indices[col + 1][r] = sgn * cluster.layer_table_indices[r][c]
            cluster.layer_table_indices = new_indices
            cluster.dims = (cluster.dims[0] + 1, cluster.dims[1] + 1)
            return True

        def lower_matrix_sum(cluster: Cluster) -> bool:
            if any(i == 1 for i in cluster.dims):
                return False
            for i in range(cluster.dims[0]):
                if len(cluster.layer_table_indices[i]) != i + 1:
                    return False
            sz = len(cluster.layer_table_indices) + 1
            new_indices: List[List[Optional[int]]] = [[None for _ in range(sz)] for _ in range(sz)]
            sgn = 1
            if cluster.function_type == "mix":
                sgn = -1
            for r in range(sz - 1):
                for c in range(len(cluster.layer_table_indices[r])):
                    new_indices[r][c] = sgn * cluster.layer_table_indices[r][c]
                    new_indices[c][r] = sgn * cluster.layer_table_indices[r][c]
            cluster.layer_table_indices = new_indices
            cluster.dims = (cluster.dims[0] + 1, cluster.dims[1] + 1)
            return True

        def matrix_sum(cluster: Cluster) -> bool:
            if any(i == 1 for i in cluster.dims):
                return False
            unique_indices = [list(x) for x in set(tuple(x) for x in cluster.layer_table_indices)]
            if len(unique_indices) == len(cluster.layer_table_indices):
                return True
            return False

        def row_sum(cluster: Cluster) -> bool:
            first_row_index = cluster.layer_table_indices[0][0]
            same_row = True
            for index in cluster.layer_table_indices:
                if first_row_index != index[0]:
                    same_row = False
            if not same_row:
                return False
            if len(cluster.layer_table_indices) == cluster.dims[1]:
                unique_indices = [list(x) for x in set(tuple(x) for x in cluster.layer_table_indices)]
                if len(unique_indices) == len(cluster.layer_table_indices):
                    return True
            return False

        def col_sum(cluster: Cluster) -> bool:
            first_col_index = cluster.layer_table_indices[0][1]
            same_col = True
            for index in cluster.layer_table_indices:
                if first_col_index != index[1]:
                    same_col = False
            if not same_col:
                return False
            if len(cluster.layer_table_indices) == cluster.dims[0]:
                unique_indices = [list(x) for x in set(tuple(x) for x in cluster.layer_table_indices)]
                if len(unique_indices) == len(cluster.layer_table_indices):
                    return True
            return False

        def global_for_loop(cluster: Cluster) -> bool:
            if (
                any(isinstance(st, dict) for st in cluster.starts)
                and len(utils.get_all_elements(cluster.layer_table_indices)) > 1
            ):
                return True
            return False

        def get_global_string(input: Dict[str, Any], is_start: bool = False) -> str:
            input_type = input["type"]
            pattern = input["param"]
            range_string = ""
            if input_type == "constant":
                if pattern != 0:
                    range_string += str(pattern)
                    if is_start:
                        range_string += ", "
            elif input_type == "variable":
                it_sym_num = 0
                for j in range(len(pattern) - 1):
                    if pattern[j] == 0:
                        continue
                    if it_sym_num > 0 and pattern[j] > 0:
                        range_string += "+"
                    elif it_sym_num > 0 and pattern[j] < 0:
                        range_string += "-"
                    if abs(pattern[j]) != 1:
                        range_string += str(abs(pattern[j])) + "*"
                    range_string += self.iterator_symbols[it_sym_num]
                    it_sym_num += 1

                if pattern[-1] != 0:
                    if pattern[-1] > 0:
                        range_string += "+"
                    range_string += str(pattern[-1])

                if is_start:
                    range_string += ", "

            return range_string

        all_iterator_symbols = ["i", "j", "k", "l", "m", "n"]
        if global_for_loop(self.cluster) == True:
            self.loop_pattern = "global"
            self.fixed_index = None
            count = 0
            self.iterator_symbols = []
            for st in self.cluster.starts:
                if st["type"] != "skip" and st["type"] is not None:
                    self.iterator_symbols.append(all_iterator_symbols[count])
                    count += 1
            self.num_iterators = count
            iterator_string = ""
            for i in range(self.num_iterators):
                if i > 0:
                    iterator_string += ","
                iterator_string += self.iterator_symbols[i]
            if self.num_iterators == 0:
                self.assignment_string = self.cluster.layer_symbol
            else:
                self.assignment_string = self.cluster.layer_symbol + f"[{iterator_string}]"
            self.loop_string = "\t"
            for i, s_i in enumerate(
                [j for j, st in enumerate(self.cluster.starts) if not (st["type"] is None or st["type"] == "skip")]
            ):
                self.loop_string += f"for {self.iterator_symbols[i]} in range("
                self.loop_string += get_global_string(self.cluster.starts[s_i], True)
                self.loop_string += get_global_string(self.cluster.stops[s_i], False)
                self.loop_string += "):\r\n"

                if self.cluster.starts[s_i].get("symmetric_zero", False):
                    self.loop_string += "\t" * (i + 2)
                    self.loop_string += (
                        f"if {self.iterator_symbols[i-1]} == {self.iterator_symbols[i]}:\r\n"
                        + ("\t" * (i + 3))
                        + "continue\r\n"
                    )

                self.loop_string += "\t" * (i + 2)
            self.loop_string += self.place_holder + "\r\n"
        elif stand_alone(self.cluster) == True:
            self.loop_pattern = "alone"
            self.fixed_index = None
            self.num_iterators = 0
            self.assignment_string = self.cluster.layer_symbol
            self.loop_string = "\t" + self.place_holder + "\r\n"
        elif one_dim_sum(self.cluster) == True:
            self.loop_pattern = "1d_sum"
            self.fixed_index = None
            self.num_iterators = 1
            self.iterator_symbols = self.iterator_symbols[: self.num_iterators]
            self.assignment_string = self.cluster.layer_symbol + f"[{self.iterator_symbols[0]}]"
            range_end = [i for i in self.cluster.dims if i > 1][0]
            self.loop_string = (
                f"\tfor {self.iterator_symbols[0]} in range({range_end}):\r\n\t\t" + self.place_holder + "\r\n"
            )
        elif upper_matrix_sum(self.cluster) == True:
            self.loop_pattern = "upper_mat"
            self.fixed_index = None
            self.num_iterators = 2
            self.iterator_symbols = self.iterator_symbols[: self.num_iterators]
            self.assignment_string = (
                self.cluster.layer_symbol + f"[{self.iterator_symbols[0]},{self.iterator_symbols[1]}]"
            )
            self.loop_string = (
                f"\tfor {self.iterator_symbols[0]} in range({self.cluster.dims[0]}):"
                + f"\r\n\t\tfor {self.iterator_symbols[1]} in range({self.cluster.dims[1]}):"
                + f"\r\n\t\t\tif {self.iterator_symbols[0]}=={self.iterator_symbols[1]}:\r\n\t\t\t\tcontinue\r\n\t\t\t"
                + self.place_holder
                + "\r\n"
            )
        elif lower_matrix_sum(self.cluster) == True:
            self.loop_pattern = "lower_mat"
            self.fixed_index = None
            self.num_iterators = 2
            self.iterator_symbols = self.iterator_symbols[: self.num_iterators]
            self.assignment_string = (
                self.cluster.layer_symbol + f"[{self.iterator_symbols[0]},{self.iterator_symbols[1]}]"
            )
            self.loop_string = (
                f"\tfor {self.iterator_symbols[0]} in range({self.cluster.dims[0]}):"
                + f"\r\n\t\tfor {self.iterator_symbols[1]} in range(0, {self.iterator_symbols[0]}-1):\r\n\t\t\t"
                + self.place_holder
                + "\r\n"
            )
        elif matrix_sum(self.cluster) == True:
            self.loop_pattern = "mat"
            self.fixed_index = None
            self.num_iterators = 2
            self.iterator_symbols = self.iterator_symbols[: self.num_iterators]
            self.assignment_string = (
                self.cluster.layer_symbol + f"[{self.iterator_symbols[0]},{self.iterator_symbols[1]}]"
            )
            self.loop_string = (
                f"\tfor {self.iterator_symbols[0]} in range({self.cluster.dims[0]}):"
                + f"\r\n\t\tfor {self.iterator_symbols[1]} in range({self.cluster.dims[1]}):\r\n\t\t\t"
                + self.place_holder
                + "\r\n"
            )
        elif row_sum(self.cluster) == True:
            self.loop_pattern = "row"
            self.fixed_index = self.cluster.layer_table_indices[0][0]
            self.num_iterators = 1
            self.iterator_symbols = self.iterator_symbols[: self.num_iterators]
            self.assignment_string = self.cluster.layer_symbol + f"[{self.fixed_index},{self.iterator_symbols[0]}]"
            self.loop_string = (
                f"\tfor {self.iterator_symbols[0]} in range({self.cluster.dims[1]}):\r\n\t\t"
                + self.place_holder
                + "\r\n"
            )
        elif col_sum(self.cluster) == True:
            self.loop_pattern = "col"
            self.fixed_index = self.cluster.layer_table_indices[0][1]
            self.num_iterators = 1
            self.iterator_symbols = self.iterator_symbols[: self.num_iterators]
            self.assignment_string = self.cluster.layer_symbol + f"[{self.iterator_symbols[0]},{self.fixed_index}]"
            self.loop_string = (
                f"\tfor {self.iterator_symbols[0]} in range({self.cluster.dims[0]}):\r\n\t\t"
                + self.place_holder
                + "\r\n"
            )

        if self.loop_pattern is None:
            raise ValueError(f"Cluster {self.cluster.cluster_index} has no spatially identifiable loop pattern")

    def set_function_string(self) -> None:
        def get_global_op(cluster: Cluster, num_tabs: int) -> str:
            weights: List[float | int] = []
            num_el: List[int] = []
            bias = cluster.bias
            tabs = ""
            for _ in range(num_tabs):
                tabs += "\t"
            for f in cluster.parent_neuron.functions_per_weight:
                weights.append(f.weights[0])
                num_el.append(f.num_indices)
            kw_pairs = [kw for kw in cluster.key_weight_pairs]

            order = np.argsort([-w for w in weights])
            kw_pairs = [kw_pairs[i] for i in order]
            weights = [weights[i] for i in order]
            num_el = [num_el[i] for i in order]

            for i in range(len(kw_pairs)):
                kw_pair: List[Any] = list(kw_pairs[i])
                kw_pair[1] = [1]
                kw_pairs[i] = tuple(kw_pair)
            fns = []

            fn = ""

            for k, kw in enumerate(kw_pairs):
                found_simple = False
                if len(cluster.original_neuron_ids) == 1:
                    if isinstance(kw[2], int):
                        if kw[2] >= 0:
                            found_simple = True
                    elif len(kw[2]) == 1:
                        found_simple = True
                if found_simple:
                    f = cluster.parent_neuron.input_symbol
                    f += str(kw[3])
                    input_num = kw[3]
                    if input_num == "":
                        input_num = 1
                    else:
                        input_num = int(input_num)
                    if cluster.parent_neuron.input_length[input_num - 1] > 1:
                        f += "[" + str(kw[2]) + "]"
                    fns.append(f)
                    found_simple = True
                else:
                    f = get_op(kw, order[k])
                    if kw[0] == "part_sum":
                        fn += f"part_sum = {f}\r\n\t" + tabs
                        f = "part_sum"
                    fns.append(f)

            altfns = []
            printalt = []
            for k, kw in enumerate(kw_pairs):
                if len(kw[0]) > 0:
                    if kw[0] == "value":
                        altfns.append(fns[k])
                        printalt.append(False)
                    else:
                        altfns.append(kw[0])
                        printalt.append(True)
                else:
                    altfns.append("sum" + kw[3])
                    printalt.append(True)

            # Make sure each variable name is unique
            checked = [False for _ in altfns]
            for i in range(len(altfns)):
                if checked[i]:
                    continue
                matched_ins = [j for j, f in enumerate(altfns) if f == altfns[i]]
                if len(matched_ins) == 1:
                    continue
                for j, k in enumerate(matched_ins):
                    altfns[k] += "_" + str(j + 1)
                    checked[k] = True

            cases = get_mult_cases(weights, bias, num_el)
            if cases is not None:
                self.signs = False
                case_ifs = []
                if self.previous_signs:
                    fns = [f + ">0" for f in fns]
                for case in cases:
                    case = [case[i] for i in np.argsort(np.sum(np.abs(case), axis=1))]
                    f: str = ""
                    for c_i, c in enumerate(case):
                        if c_i > 0:
                            f += " or "
                        num = np.sum(np.abs(c))
                        if num > 1:
                            f += "("
                        count = 0
                        for j in range(len(c)):
                            if c[j] == 0:
                                continue
                            if count > 0:
                                f += " and "
                            count += 1
                            if c[j] > 0:
                                f += fns[j]
                            else:
                                f += "(not " + fns[j] + ")"

                        if count == 0:
                            f += "True"
                        if count > 1:
                            f += ")"
                    # f += ":"
                    # f += "\r\n\t\t" + tabs
                    case_ifs.append(f)
                if len(case_ifs) == 1:
                    fn += self.assignment_string + " = " + case_ifs[0]
                    fn += "\r\n\t" + tabs
                    self.initialize = self.loop_pattern != "alone"
                else:
                    fn += "if " + case_ifs[0] + ":\r\n\t\t" + tabs
                    fn += self.assignment_string + " = 1"
                    fn += "\r\n\t" + tabs
                    fn += "elif " + case_ifs[1] + ":\r\n\t\t" + tabs
                    fn += self.assignment_string + " = -1"
                    fn += "\r\n\t" + tabs
                return fn

            if not self.act:
                return sum_function(weights, bias, fns)

            all_fn = check_all(weights, bias, num_el, fns, tabs)
            if all_fn is not None:
                return all_fn
            any_fn = check_any(weights, bias, num_el, fns, tabs)
            if any_fn is not None:
                return any_fn

            if len(weights) == 1:
                boundary = -bias / weights[0]
                if np.abs(boundary) < 1e-6:
                    boundary = 0
                fn += "if "
                fn += fns[0]
                if boundary == round(boundary):
                    fn += " > " + str(boundary)
                else:
                    fn += " > " + str(math.floor(boundary))
                fn += ":\r\n\t\t" + tabs

                fn += self.assignment_string + " = 1\r\n\t" + tabs

                fn += "elif "
                fn += fns[0]
                if boundary == round(boundary):
                    fn += " < " + str(boundary)
                else:
                    fn += " <= " + str(math.floor(boundary))
                fn += ":\r\n\t\t" + tabs

                fn += self.assignment_string + " = -1\r\n\t" + tabs
                return fn

            elif len(weights) > 2:
                bounds, edges = get_bounds(weights, bias, num_el)
                weights, bias, num_el, fns, altfns = check_means(weights, bias, num_el, fns, altfns)
                for i in range(len(weights)):
                    if printalt[i]:
                        fn += f"{altfns[i]} = {fns[i]}\r\n\t" + tabs
                fn += get_fn_if_statements(weights, bias, altfns, tabs)
            else:  # len(weights)==2
                bounds, edges = get_bounds(weights, bias, num_el)
                started = False
                for i in range(len(weights)):
                    if printalt[i]:
                        fn += f"{altfns[i]} = {fns[i]}\r\n\t" + tabs

                for e in edges:
                    if started:
                        fn += "elif "
                    else:
                        fn += "if "
                    started = True
                    fn += altfns[e[0]]
                    if e[2]:
                        fn += " > "
                    else:
                        fn += " < "
                    fn += str(e[1])
                    fn += ":\r\n\t\t" + tabs
                    fn += self.assignment_string + " = "
                    if e[3]:
                        fn += "1"
                    else:
                        fn += "-1"
                    fn += "\r\n\t" + tabs

                cases = get_cases(weights, bias, bounds)

                if not cases or (bounds[0][1] - bounds[0][0]) == (bounds[1][1] - bounds[1][0]):
                    if_statements = get_fn_if_statements(weights, bias, altfns, tabs)
                    if started:
                        if_statements = "el" + if_statements
                    fn += if_statements
                else:
                    for case in cases:
                        if started:
                            add_el = "el"
                        else:
                            add_el = ""
                        started = True
                        fn += f"{add_el}if {altfns[case[0]]} == {case[1]}:\r\n\t\t"

                        if abs(case[2]) != num_el[1 - case[0]] - 1:
                            if_statement = f"if {altfns[1-case[0]]} GSIGN {case[2]}"
                            assign_statement = f":\r\n\t\t\t{tabs}{self.assignment_string} = ASSIGN\r\n\t{tabs}"
                            sgns = [">", "<"]
                        elif (
                            case[2] > 0
                            and kw_pairs[1 - case[0]][0] != "offcol_sum"
                            and kw_pairs[1 - case[0]][0] != "offrow_sum"
                        ):
                            altf = (fns[1 - case[0]].replace("np.sum(", ""))[:-1]
                            if_statement = f"ifGSIGN np.all({altf}==1)"
                            assign_statement = ":\r\n\t\t\t{tabs}{self.assignment_string} = ASSIGN\r\n\t{tabs}"
                            sgns = ["", " not"]
                        elif (
                            case[2] < 0
                            and kw_pairs[1 - case[0]][0] != "offcol_sum"
                            and kw_pairs[1 - case[0]][0] != "offrow_sum"
                        ):
                            altf = (fns[1 - case[0]].replace("np.sum(", ""))[:-1]
                            if_statement = f"ifGSIGN np.all({altf}==-1)"
                            assign_statement = f":\r\n\t\t\t{tabs}{self.assignment_string} = ASSIGN\r\n\t{tabs}"
                            sgns = [" not", ""]

                        for i in range(2):
                            if i == 0 or round(case[2], 0) == case[2]:
                                statement = if_statement.replace("GSIGN", sgns[i])
                                if i > 0:
                                    statement = "el" + statement
                            else:
                                statement = "else"
                                self.initialize = False
                                self.signs = False

                            if case[3]:
                                statement += assign_statement.replace("ASSIGN", str(int(1 - 2 * i)))
                            else:
                                statement += assign_statement.replace("ASSIGN", str(int(2 * i - 1)))

                            if i == 0:
                                statement += "\t"
                            fn += statement

            return fn

        def check_means(
            w: List[float | int], b: float, lims: List[float], fns: List[str], altfns: List[str]
        ) -> Tuple[List[float | int], float, List[float], List[str], List[str]]:
            new_weights = [w[i] * lims[i] for i in range(len(w))]
            mn = min(abs(n) for n in new_weights)
            if mn > 1:
                temp_weights = [new_weights[i] / mn for i in range(len(w))]
                if all(abs(t - round(t)) < 1e-4 for t in temp_weights):
                    lims = [lims[i] / (new_weights[i] / w[i]) for i in range(len(lims))]
                    w = temp_weights
                    b /= mn
                    fns = [f.replace("sum", "mean") for f in fns]
                    altfns = [f.replace("sum", "mean") for f in altfns]
            return w, b, lims, fns, altfns

        def check_all(
            w: List[int | float], b: int | float, lims: List[int | float], fns: List[str], tabs: str
        ) -> Optional[str]:
            n = len(w)
            for i in range(2**n):
                sgns = [int((i % 2 ** (j + 1)) < (2**j)) for j in range(n)]
                if self.previous_signs:
                    sgns = [s * 2 - 1 for s in sgns]
                all_sum = sum(sgns[j] * w[j] * lims[j] for j in range(n)) + b
                off_sums = []
                for j in range(n):
                    off_sums.append(np.sign(all_sum - sgns[j] * w[j]))
                all_sum = np.sign(all_sum)
                if any(all_sum == o for o in off_sums):
                    continue
                if any((off_sums[0] != o or o == 0) for o in off_sums):
                    continue
                fn = "if "
                for j in range(n):
                    if j > 0:
                        fn += " and "
                    fns[j] = fns[j].replace("np.sum(", "")
                    fns[j] = fns[j].replace(")", "")
                    fn += f"np.all({fns[j]}=={sgns[j]})"
                fn += ":\r\n\t\t" + tabs
                fn += f"{self.assignment_string} = {int(all_sum)}\r\n\t" + tabs
                fn += "else:\r\n\t\t" + tabs + f"{self.assignment_string} = {int(off_sums[0])}\r\n"
                return fn

            return None

        def check_any(w, b, lims, fns, tabs):
            n = len(w)
            if n > 1:
                return None

            if self.previous_signs:
                sgns = [-1, 1]
            else:
                sgns = [0, 1]

            for s, sgn in enumerate(sgns):
                sgn1 = np.sign(w[0] * lims[0] * sgn + b)
                sgn2 = np.sign(w[0] * (lims[0] - 1) * sgn + b + w[0] * sgns[1 - s])
                if sgn1 * sgn2 < 0:
                    fns[0] = fns[0].replace("np.sum(", "")
                    fns[0] = fns[0].replace(")", "")
                    fn = f"if np.any({fns[0]}!={sgn}):\r\n\t\t{tabs}"
                    fn += f"{self.assignment_string} = {sgn2}\r\n\t{tabs}"
                    fn += f"else:\r\n\t\t{tabs}{self.assignment_string} = {sgn1}\r\n"
                    return fn

            return None

        def get_fn_if_statements(w, b, fns, tabs):
            if b == 0:
                return get_simpler_if_statements(w, b, fns, tabs)

            w_strings = []
            for i in range(len(w)):
                if i == 0:
                    w_strings.append("")
                elif w[i] > 0:
                    w_strings.append(" + ")
                else:
                    w_strings.append(" - ")

                if i == 0 and w[i] == -1:
                    w_strings[-1] = "-"
                elif abs(w[i]) != 1:
                    w_strings[-1] += str(abs(w[i])) + "*"

            sgns = [">", "<"]

            compare_string = "if "

            for i in range(len(w)):
                compare_string += w_strings[i]
                compare_string += fns[i]
            use_else = False
            if b != int(b):
                b = -math.ceil(b)
                use_else = True
            compare_string += " GLSIGN " + str(-b) + ":"
            assign_string = "\r\n\t\t" + tabs + self.assignment_string + " = ASSIGN\r\n\t" + tabs

            fn = ""
            for i in range(2):
                add_string = compare_string.replace("GLSIGN", sgns[i])
                fn += add_string + assign_string.replace("ASSIGN", str(int(1 - 2 * i)))
                if not use_else:
                    compare_string = "el" + compare_string
                else:
                    compare_string = "else:"
            return fn

        def get_simpler_if_statements(w: List[float], b: float, fns: List[str], tabs: List[str]) -> str:
            ind = np.where(np.abs(w) == np.min(np.abs(w)))[0][0]

            w0 = w[ind]
            fn0 = fns[ind]
            del w[ind]
            del fns[ind]
            w = [-i for i in w]

            w_strings = []
            for i in range(len(w)):
                if i == 0:
                    w_strings.append("")
                elif w[i] > 0:
                    w_strings.append(" + ")
                else:
                    w_strings.append(" - ")

                if i == 0 and w[i] < 0:
                    w_strings[-1] = "-"
                if abs(w[i]) != 1:
                    w_strings[-1] += str(abs(w[i])) + "*"

            w0_string = ""
            if w0 < 0:
                w0_string += "-"
            if abs(w0) != 1:
                w0_string += str(abs(w0)) + "*"

            compare_string = f"if {w0_string}{fn0} GLSIGN "
            for i in range(len(w)):
                compare_string += w_strings[i]
                compare_string += fns[i]
            compare_string += f":\r\n\t\t{tabs}{self.assignment_string} = ASSIGN\r\n\t{tabs}"

            sgns = [">", "<"]
            fn = ""
            for i in range(2):
                add_string = compare_string.replace("GLSIGN", sgns[i])
                fn += add_string.replace("ASSIGN", str(int(1 - 2 * i)))
                if i == 0:
                    compare_string = "el" + compare_string
            return fn

        def get_mult_cases(w: float, b: float, num_el: List[int]) -> Optional[List[List[List[int]]]]:
            if any(n > 1 for n in num_el):
                return None

            n = len(w)
            truth_table = np.zeros((2**n, n))
            for i in range(n):
                truth_table[:, -(i + 1)] = ((np.arange(2**n) % (2 ** (i + 1))) >= (2**i)).astype(int)

            if self.symbolic_inputs:
                truth_table *= 2
                truth_table -= 1

            sums = np.ones(len(truth_table)) * b
            for i in range(len(w)):
                sums += w[i] * truth_table[:, i]
            labels = np.sign(sums)
            if any(labels == 0):
                label_sets = [np.sign(labels - 0.5), np.sign(labels + 0.5)]
            else:
                label_sets = [labels]

            if not self.symbolic_inputs:
                truth_table *= 2
                truth_table -= 1
            all_trues = []
            for labels in label_sets:
                trues = []
                for l in range(len(labels)):
                    if labels[l] > 0:
                        trues.append((truth_table[l, :]).astype(int).tolist())

                while True:
                    found_something = False
                    for i in range(len(trues)):
                        for j in range(len(trues[i])):
                            if trues[i][j] == 0:
                                continue
                            temp_true = trues[i].copy()
                            temp_true[j] *= -1
                            if temp_true in trues:
                                found_something = True
                                del trues[i]
                                del trues[[k for k, t in enumerate(trues) if t == temp_true][0]]
                                temp_true[j] = 0
                                trues.append(temp_true)
                                break
                        if found_something:
                            break
                    if not found_something:
                        break

                order = np.argsort(np.sum(np.abs(np.array(trues)), axis=1))
                trues = [trues[o] for o in order]

                for t in range(len(trues)):
                    for t2 in range(t + 1, len(trues)):
                        matched = True
                        inds = []
                        for i in range(len(trues[t])):
                            if trues[t][i] == 0:
                                continue
                            if trues[t][i] != -trues[t2][i]:
                                matched = False
                                break
                            inds.append(i)
                        if matched:
                            for i in inds:
                                trues[t2][i] = 0

                all_trues.append(trues)

            return all_trues

        def get_cases(
            w: List[float], b: float, bounds: List[float]
        ) -> Optional[List[Tuple[int, int, int | float, bool]]]:
            lims = [bound[1] - bound[0] + 1 for bound in bounds]
            ind = 0 if lims[1] >= lims[0] else 1
            if lims[ind] == 1:
                ind = 1 - ind
            off_ind = 1 - ind

            if lims[ind] > 5:
                return None

            cases = []  # dimension, fixed, boundary, + or - assignment for >
            for fixed in range(bounds[ind][1], bounds[ind][0] - 1, -1):
                boundary = -(b + w[ind] * fixed) / w[off_ind]
                if not (boundary < bounds[off_ind][1] and boundary > bounds[off_ind][0]):
                    continue
                if abs(int(boundary) - boundary) < 1e-4:
                    boundary = int(boundary)
                v = w[ind] * fixed + w[off_ind] * (boundary + 1) + b
                cases.append((ind, fixed, boundary, v > 0))

            return cases

        def get_bounds(w, b, a, val=-1):
            z0 = [0, 0]
            z1 = [0, 0]
            z0[0] = -(b + val * w[1] * a[1]) / w[0]
            z0[1] = -(b + w[1] * a[1]) / w[0]
            z1[0] = -(b + val * w[0] * a[0]) / w[1]
            z1[1] = -(b + w[0] * a[0]) / w[1]

            edges = []  # dimension, boundary, > or < boundary, + or - assignment
            bounds = [[-abs(a[0]), abs(a[0])], [-abs(a[1]), abs(a[1])]]
            for i, z in enumerate([z0, z1]):
                if abs(z[0]) <= a[i] - 1:
                    if z[0] > z[1]:
                        edges.append((i, math.floor(z[0]), True, w[i] > 0))
                        bounds[i][1] = math.floor(z[0])
                    else:
                        edges.append((i, math.ceil(z[0]), False, w[i] < 0))
                        bounds[i][0] = math.ceil(z[0])
                if abs(z[1]) <= a[i] - 1:
                    if z[1] > z[0]:
                        edges.append((i, math.floor(z[1]), True, w[i] > 0))
                        bounds[i][1] = math.floor(z[1])
                    else:
                        edges.append((i, math.ceil(z[1]), False, w[i] < 0))
                        bounds[i][0] = math.ceil(z[1])

            return bounds, edges

        def get_op(kw_pair: List, id: int, do_abs: bool = False) -> str:
            result = ""
            key, weight = kw_pair[0], kw_pair[1]
            in_sym = self.cluster.parent_neuron.input_symbol
            in_sym += kw_pair[3]
            if do_abs:
                w = int(abs(weight[0]))
                w_str = str(w) + "*"
                if w == 1:
                    w_str = ""
            else:
                w = int(weight[0])
                w_str = str(w) + "*"
                if w == 1:
                    w_str = ""
                elif w == -1:
                    w_str = "-"

            op_iterators = []
            for i, f in enumerate(self.cluster.function_ids):
                if f == id:
                    it = self.cluster.iterator_inds[i]
                    if it is not None:
                        op_iterators.append(self.iterator_symbols[it])
                    elif self.cluster.starts[i]["type"] is None:
                        op_iterators.append(str(self.cluster.starts[i]["param"]))

            if key == "full_sum" or key == "mat_sum":
                res = w_str + f"np.sum({in_sym})"
            elif key == "value":
                if (not isinstance(kw_pair[2], (list, tuple))) or len(kw_pair[2]) == 1:
                    res = f"{w_str}{in_sym}[{op_iterators[0]}]"
                else:
                    res = f"{w_str}{in_sym}[{op_iterators[0]}, {op_iterators[1]}]"
            elif key == "part_sum":
                sum_string = ""
                for i, s in enumerate(kw_pair[2]):
                    if i > 0:
                        sum_string += " + "
                    if isinstance(s, int) or len(s) == 1:
                        sum_string += f"{in_sym}[{s}]"
                    else:
                        sum_string += f"{in_sym}[{s[0]},{s[1]}]"
                res = w_str + "({sum_string})"
            elif key == "partial_sum":
                res = w_str
                # NEED TO FINISH
            elif key == "row_sum":
                if kw_pair[4] is None:
                    filler = ":"
                else:
                    if kw_pair[4] == [i for i in range(max(kw_pair[4]) + 1)]:
                        filler = str(max(kw_pair[4]) + 1) + ":"
                    else:
                        raise ValueError("Cannot use these missing indices")
                res = w_str + "np.sum({input_symbol}[{iterator}, {filler}])"
                res = res.format(input_symbol=in_sym, iterator=op_iterators[0], filler=filler)
            elif key == "col_sum":
                if kw_pair[4] is None:
                    filler = ":"
                else:
                    if kw_pair[4] == [i for i in range(max(kw_pair[4]) + 1)]:
                        filler = str(max(kw_pair[4]) + 1) + ":"
                    else:
                        raise ValueError("Cannot use these missing indices")
                res = w_str + "np.sum({input_symbol}[{filler}, {iterator}])"
                res = res.format(input_symbol=in_sym, iterator=op_iterators[0], filler=filler)
            elif key == "offrow_sum":
                res = w_str + "(np.sum({input_symbol}) - np.sum({input_symbol}[{iterator}, :]))"
                res = res.format(input_symbol=in_sym, iterator=op_iterators[0])
            elif key == "offcol_sum":
                res = w_str + "(np.sum({input_symbol}) - np.sum({input_symbol}[:, {iterator}]))"
                res = res.format(input_symbol=in_sym, iterator=op_iterators[0])
            elif key == "off_sum":
                res = w_str + "(np.sum({input_symbol}) - np.sum({input_symbol}[{iterator}]))"
                res = res.format(input_symbol=in_sym, iterator=op_iterators[0])
            else:
                raise ValueError(f"The given key value, {key}, is not a valid key")
            result = res
            return result

        def sum_function(w, b, f):
            fn_string = self.assignment_string + " = "

            order = np.argsort(w)[::-1]
            for j, i in enumerate(order):
                w_string = ""
                if j > 0:
                    if w[i] > 0:
                        w_string = " + "
                    else:
                        w_string = " - "
                elif w[i] < 0:
                    w_string = "-"
                if abs(w[i]) != 1:
                    w_string += str(abs(w[i])) + "*"
                fn_string += w_string + f[i]

            return fn_string

        # BEGIN FUNCTION HERE TO WRITE THE FUNCTION STRING
        fn = get_global_op(self.cluster, self.num_iterators)
        if fn is None:
            fn = ""
            for index, kw_pair in enumerate(self.cluster.key_weight_pairs):
                if index > 0:
                    if kw_pair[1][0] < 0:
                        fn += " - "
                    else:
                        fn += " + "
                fn_string = get_op(kw_pair, index, do_abs=index == 0)
                fn += fn_string
            if self.cluster.bias > 0:
                fn += " + " + str(int(self.cluster.bias))
            elif self.cluster.bias < 0:
                fn += " - " + str(-int(self.cluster.bias))
            self.function_string = f"np.sign({fn})"
        else:
            self.function_string = fn
            self.global_found = True

    def write_code(self, filename: str, initialization: str) -> None:
        if self.global_found:
            inner = self.function_string
        else:
            inner = self.assignment_string + " = " + self.function_string
        if not self.signs:
            inner = inner.replace("= 0", "= 0.5")
            inner = inner.replace("= -1", "= 0")
        self.loop_string = self.loop_string.replace("INSIDE_LOOP", inner)
        self.loop_string = self.loop_string.replace("\r\n", "\n")
        with open(filename, "a") as f:
            if self.initialize:
                f.write(initialization)
            f.write(self.loop_string)
