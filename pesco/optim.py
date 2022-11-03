import numpy as np
import time

from collections import namedtuple

from ortools.linear_solver import pywraplp

def optimize_portfolio(solve_labels, runtimes, penalty=None, max_runtime = 900, q = 60):
    if penalty is None: penalty = max_runtime + 1
    schedule = _optimize_schedule(solve_labels, runtimes, penalty, max_runtime, q)
    sequence = compute_optimal_permutation(schedule, solve_labels, runtimes, max_runtime)

    return sequence


# Scheduling ----------------------------------------------------------------

Solver = namedtuple('Solver', ["tool", "time"])
OptimalSolver = namedtuple('OptimalSolver', ["portfolio", "unsolved", "runtime"])

# Compute sparse solver sets
def _compute_solver_sets(solve_labels, runtimes, max_runtime = 900, q = 60):
    num_steps   = max_runtime // q
    solver_sets = np.zeros((num_steps,) + solve_labels.shape)

    for i in range(num_steps):
        timelimit = q * (i + 1)
        timemask  = np.clip(timelimit - runtimes, 0, 1)
        solver_sets[i] = solve_labels * timemask

    # Sparsify
    def _sparse_solvers(solvers):
        return set(Solver(j, q * (i + 1)) for i, j in zip(*solvers.nonzero()))

    solver_sets = {i: _sparse_solvers(solver_sets[:, i]) for i in range(solver_sets.shape[1])}
    return {k: v for k, v in solver_sets.items() if len(v) > 0}


def _optimize_schedule(solve_labels, runtimes, penalty = 901, max_runtime = 900, q = 60):
    
    solver_sets = _compute_solver_sets(solve_labels, runtimes, max_runtime, q)
    solvers = list(set.union(*[s for s in solver_sets.values()]))
    solvables = list(solver_sets.keys())
    non_solvable = [i for i in range(solve_labels.shape[0]) if i not in solver_sets]

    # Build IP
    ip_solver = pywraplp.Solver.CreateSolver('SCIP')

    # Variables that mark instances as unsolvable
    unsolvables = [ip_solver.BoolVar("unolvable_%d" % i)  for i in solvables]

    # Variable mark active solver in schedule
    active_solvers = [ip_solver.BoolVar(f"{sol[0]}_{sol[1]}") for sol in solvers]

    # Each task is either solved or marked as unsolvable
    for i, solvable in enumerate(solvables):
        unsolvable = unsolvables[i]
        solver_set = solver_sets[solvable]
        ip_solver.Add(
            unsolvable + sum(active_solver for i, active_solver in enumerate(active_solvers) if solvers[i] in solver_set) >= 1
        )

    # All solvers cannot take more time than the maximal time
    total_runtime = sum( solvers[i].time * active_solver for i, active_solver in enumerate(active_solvers))
    ip_solver.Add(total_runtime <= max_runtime)

    # Minimize penalty over full set
    if isinstance(penalty, int):
        unsolvable_penalty = (penalty + 1) * sum(unsolvables)
    else:
        unsolvable_penalty = sum(penalty[s] * unsolvables[i] for i, s in enumerate(solvables))

    ip_solver.Minimize(unsolvable_penalty + total_runtime)

    # Solve problem
    status = ip_solver.Solve()

    # If an optimal solution has been found, print results
    if status == pywraplp.Solver.OPTIMAL:
        optimal_solver = {}
        for i, active_solver in enumerate(active_solvers):
            if active_solver.solution_value() == 1:
                solver, time = solvers[i]
                optimal_solver[solver] = time

        return OptimalSolver(
            optimal_solver,
            non_solvable + \
            [solvables[i] for i, unsolvable in enumerate(unsolvables) if unsolvable.solution_value()],
            ip_solver.wall_time()
        )

    else:
        print('The solver could not find an optimal solution.')
        return None


# Compute optimal permutation ------------------------------------------------

def compute_optimal_permutation(schedule, solve_labels, runtimes, max_runtime):
    if len(schedule.portfolio) == 0:
        return schedule

    if len(schedule.unsolved) == 0:
        portfolio = sorted(schedule.portfolio.items(), key=lambda x: x[1])
        portfolio[-1] = (portfolio[-1][0], 900)
        return OptimalSolver(tuple(portfolio), schedule.unsolved, schedule.runtime)
    
    start_time = time.time()

    # Reduce to unsolved instances (it is enough since we only expand runtime)
    unsolved     = np.array(schedule.unsolved)
    solve_labels = solve_labels[unsolved]
    runtimes     = runtimes[unsolved]

    # Identify instances where a solver fails inside the time bound
    time_mask = np.zeros(runtimes.shape[1])
    for p, runtime in schedule.portfolio.items():
        time_mask[p] = runtime

    time_bitmask = np.clip(time_mask - runtimes, 0, 1)
    mask_runtimes = time_bitmask * runtimes + (1 - time_bitmask) * time_mask
    cum_mask_runtimes = mask_runtimes.sum(axis = 1)
    
    # Test each solver
    best_solver_id = -1
    best_nsolved   = -1
    best_solved    = None

    for solver_id in schedule.portfolio.keys():
        used_time  = cum_mask_runtimes - mask_runtimes[:, solver_id]
        avail_time = max_runtime - used_time
        time_mask  = np.clip(runtimes[:, solver_id] - avail_time, 0, 1)
        newly_solved = time_mask * solve_labels[:, solver_id]
        newly_score  = newly_solved.sum()
        if newly_score > best_nsolved:
            best_nsolved = newly_score
            best_solved  = newly_solved
            best_solver_id = solver_id

    # Build sequence
    sequence = [(k, v) for k, v in schedule.portfolio.items() if k != best_solver_id]
    sequence = sorted(sequence, key=lambda x: x[1])
    sequence = tuple(sequence) + ((best_solver_id, max_runtime),)
    
    unsolved = [n for i, n in enumerate(schedule.unsolved) if best_solved[i] == 0]
    wall_time = 1000 * (time.time() - start_time) + schedule.runtime

    return OptimalSolver(sequence, unsolved, wall_time)


# Portfolio Boosting -------------------------------
def _default_optimizer(y, runtimes, penalty = None):
    return optimize_portfolio(y, runtimes, penalty).portfolio


class PortfolioBoosting:

    def __init__(self, selector_constructor, optimizer = None, weight = 9000, max_iter = 20, delta = 0.9, n_solver = -1, add_base = False):
        self.selector_constructor = selector_constructor
        self.optimizer = optimizer if optimizer is not None else _default_optimizer
        self.weight = weight
        self.max_iter = max_iter
        self.delta = delta
        self.n_solver = n_solver
        self.add_base = add_base

        self._tools = []
        self._selector = None

    def _evaluate(self, candidate, y, runtimes):
        if isinstance(candidate, int):
            candidate = [(candidate, 900)]

        eval_results   = np.zeros((y.shape[0],))
        eval_runtimes  = np.zeros((y.shape[0],)) 

        running = np.ones((y.shape[0],))
        for tool, timelimit in candidate:
            labels, truntimes = y[:, tool], runtimes[:, tool]
            timemask = np.clip(timelimit - truntimes + 1, 0, 1).astype(int)

            cruntimes = timemask * truntimes + (1 - timemask) * timelimit

            eval_results  = (1 - running) * eval_results + running * timemask * labels
            eval_runtimes = (1 - running) * eval_runtimes + running * (eval_runtimes + cruntimes)
            running -= running * timemask

        return eval_results, eval_runtimes


    def fit(self, X, y, runtimes):
        constructor = self.selector_constructor(X)

        weight = (y * runtimes) + ((1 - y) * self.weight)
        vbs    = weight.min(axis = 1)

        # Start regret
        regret = np.full((X.shape[0],), self.weight) - vbs

        # First iteration
        candidate = self.optimizer(y, runtimes, penalty = regret)
        candidate_results, candidate_runtimes = self._evaluate(candidate, y, runtimes)
        candidate_weight = (candidate_results * candidate_runtimes) + ((1 - candidate_results) * self.weight)

        self._tools.append(candidate)
        constructor.add(candidate_results, candidate_runtimes)
        regret = candidate_weight - vbs

        weights = [candidate_weight]
        for _ in range(self.max_iter):
            candidate = self.optimizer(y, runtimes, penalty = regret)
            candidate_results, candidate_runtimes = self._evaluate(candidate, y, runtimes)

            # Add weights ----------------------------------------
            candidate_weight = (candidate_results * candidate_runtimes) + ((1 - candidate_results) * self.weight)
            weights.append(candidate_weight)
            cweight = np.stack(weights).transpose()

            # Compute correlation -------------------------------
            covariance = np.corrcoef(cweight.transpose())
            max_cov  = covariance[-1, :-1].max()

            if max_cov > self.delta: 
                # This is already contained  / We ignore solved tasks
                regret *= (1 - candidate_results)
                weights = weights[:-1]
                continue
                
            print(max_cov, candidate)

            self._tools.append(candidate)
            constructor.add(candidate_results, candidate_runtimes)
            current_selector = constructor.build()
            selection = current_selector.predict(X)

            selected_weight = cweight[np.arange(selection.shape[0]), selection]
            regret = selected_weight - vbs
            print(regret.sum())

            if self.n_solver > 0 and len(self._tools) >= self.n_solver: break

        if self.add_base:
            for i in range(y.shape[1]):
                self._tools.append([(i, 900)])
                constructor.add(y[:, i], runtimes[:, i])

        self._selector = constructor.build()

    def predict(self, X, return_scores = False):
        assert self._selector is not None
        selection = self._selector.predict(X)
        return [self._tools[i] for i in selection]


# K-means cluster ------------

from sklearn.cluster import KMeans

class KMeansBoosting:

    def __init__(self, selector_constructor, optimizer = None, k = 4, weight = 9000, add_base = False):
        self.selector_constructor = selector_constructor
        self.optimizer = optimizer if optimizer is not None else _default_optimizer
        self.k = k
        self.weight = weight
        self.add_base = add_base

        self._tools = []
        self._selector = None

    def _evaluate(self, candidate, y, runtimes):
        if isinstance(candidate, int):
            candidate = [(candidate, 900)]

        eval_results   = np.zeros((y.shape[0],))
        eval_runtimes  = np.zeros((y.shape[0],)) 

        running = np.ones((y.shape[0],))
        for tool, timelimit in candidate:
            labels, truntimes = y[:, tool], runtimes[:, tool]
            timemask = np.clip(timelimit - truntimes + 1, 0, 1).astype(int)

            cruntimes = timemask * truntimes + (1 - timemask) * timelimit

            eval_results  = (1 - running) * eval_results + running * timemask * labels
            eval_runtimes = (1 - running) * eval_runtimes + running * (eval_runtimes + cruntimes)
            running -= running * timemask

        return eval_results, eval_runtimes


    def fit(self, X, y, runtimes):
        constructor = self.selector_constructor(X)

        cluster_assign = KMeans(n_clusters = self.k).fit_predict(X)

        for k in range(self.k):
            index = cluster_assign == k
            labels = y[index]
            truntimes = runtimes[index]

            candidate = self.optimizer(labels, truntimes)
            candidate_results, candidate_runtimes = self._evaluate(candidate, y, runtimes)
           
            self._tools.append(candidate)
            constructor.add(candidate_results, candidate_runtimes)

        if self.add_base:
            for i in range(y.shape[1]):
                self._tools.append([(i, 900)])
                constructor.add(y[:, i], runtimes[:, i])

        self._selector = constructor.build()

    def predict(self, X, return_scores = False):
        assert self._selector is not None
        selection = self._selector.predict(X)
        return [self._tools[i] for i in selection]
