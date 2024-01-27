from src.distill import condense
import numpy as np

from typing import Tuple


def main() -> None:
    problem_name = "maxsat"
    ENN_layers_filenames = [
        "./data/sat/ENN/maxsat_2359800_1.csv",
        "./data/sat/ENN/maxsat_2359800_2.csv",
        "./data/sat/ENN/maxsat_2359800_3.csv",
    ]

    input_dims = [(10, 50)]

    print("Running distill_maxsat.py")

    condense(problem_name, ENN_layers_filenames, input_dims, return_style="probability", logic_inputs=True)

    print("ENN successfully condensed. Distilled code is in output_maxsat.py")

    print("\nTesting distilled code on random Boolean formulae")
    from output_maxsat import ENN_maxsat

    test_code(ENN_maxsat, input_dims[0])
    print()

    print("\nTesting generalized distilled code (pre-condensed) on random Boolean formulae")
    from src.general_maxsat import ENN_maxsat

    def add_input_size(func, input_dims):
        def wrapper(x):
            return func(x, n=x.shape[1], m=x.shape[0])

        return wrapper

    input_dims = (20, 100)
    test_code(add_input_size(ENN_maxsat, input_dims), input_dims)
    print()


def test_code(distill_func, input_dims) -> None:
    algs = ["distilled", "approx", "greedy"]
    num_samples = 200
    num_trials = 20
    variables = input_dims[0] // 2
    num_clauses = np.unique(np.maximum(1, np.linspace(0, input_dims[1], 11)).astype(int))

    for nc in num_clauses:
        print(f"{variables} vars, {nc} clausses -   :20", end="")
        total_sat = [0, 0, 0]

        for _ in range(num_samples):
            formula = np.zeros((input_dims[1], input_dims[0]))
            for c in range(nc):
                vars = np.random.choice(np.arange(variables), 3, replace=False)
                var_indices = 2 * vars + np.random.randint(0, 2, 3)  # randomly negate them
                formula[c, var_indices] = 1
            for alg_ind, alg in enumerate(algs):
                for _ in range(num_trials):
                    temp_formula = formula.copy()
                    num_sat = 0
                    unsat = nc

                    for v in range(variables):  # Assign each variable one at a time
                        if alg == "distilled":
                            probs = distill_func(temp_formula.transpose())
                            new_x = (np.random.random() < probs[0]) * 2 - 1
                        elif alg == "greedy":
                            t = np.sum(temp_formula[:, 0])
                            f = np.sum(temp_formula[:, 1])
                            if t > f:
                                probs = [1, 0]
                            elif t < f:
                                probs = [0, 1]
                            else:
                                probs = [0.5, 0.5]
                            new_x = (np.random.random() < probs[0]) * 2 - 1
                        elif alg == "approx":
                            new_x, unsat = get_approx(temp_formula, num_sat=num_sat, unsat=unsat)

                        if new_x == -1:
                            ind = np.where(temp_formula[:, 1] == 1)[0]
                        else:
                            ind = np.where(temp_formula[:, 0] == 1)[0]
                        num_sat += len(ind)

                        # Zero out satisfied clauses
                        temp_formula[ind, :] = 0
                        if np.max(temp_formula) == 0:
                            break
                        # Move variable to the back
                        temp_formula = np.hstack((temp_formula[:, 2:], np.zeros((temp_formula.shape[0], 2))))

                    total_sat[alg_ind] += num_sat
        for alg_ind, alg in enumerate(algs):
            mean_sat = total_sat[alg_ind] / num_trials / num_samples
            print(f"{alg}: {mean_sat:.2f}", end=" ")
        print()


def get_approx(clauses, num_sat=0, unsat=0) -> Tuple[int, int]:
    # The 3/4-approximation algorithm
    cur_SAT = num_sat
    cur_UNSAT = unsat
    cur_B = 0.5 * (cur_SAT + clauses.shape[0] - cur_UNSAT)
    ft = []
    temp_UNSAT = [None, None]
    for s in range(2):
        ind = np.where(clauses[:, 1 - s])[0]
        temp_SAT = num_sat + len(ind)
        off_ind = np.setdiff1d(np.arange(len(clauses)), ind)
        temp_UNSAT[s] = len(off_ind) - np.sum(np.any(clauses[off_ind, 2:], axis=1))
        temp_B = 0.5 * (temp_SAT + clauses.shape[0] - temp_UNSAT[s])
        ft.append(temp_B - cur_B)
    f = ft[0]
    t = ft[1]
    if f < 0:
        output = [1, 0]
    elif t < 0:
        output = [0, 1]
    elif t == 0 and f == 0:
        output = [0.5, 0.5]
    else:
        output = [t / (t + f), f / (t + f)]
    assignment = int(np.random.random() < output[0])
    return assignment * 2 - 1, temp_UNSAT[assignment]


if __name__ == "__main__":
    main()
