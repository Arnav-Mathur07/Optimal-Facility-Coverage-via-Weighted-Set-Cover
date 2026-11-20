"""
Approximation and heuristic solvers for Weighted Set Cover:
1. Chvátal's Greedy Approximation Algorithm
2. Greedy + Redundancy Pruning + 1-Swap Local Search (Refined Greedy)
"""

from __future__ import annotations
import time
from typing import List, Set, Dict, Any, Tuple
import numpy as np

from .models import ProblemInstance, SolverResult


def solve_greedy_set_cover(
    instance: ProblemInstance
) -> SolverResult:
    """
    Chvátal's Greedy Set Cover Approximation Algorithm.
    
    At each step k, let U_rem be the set of currently uncovered demand zones.
    Select the facility i* that minimizes the marginal cost-per-newly-covered-zone:
        alpha_i = c_i / |S_i cap U_rem|
        
    Guaranteed approximation ratio: rho <= H_n = sum_{k=1}^n 1/k.
    Maintains an interactive trace of each selection decision.
    """
    start_time = time.perf_counter()
    n = instance.n_zones
    m = instance.n_facilities
    costs = instance.costs
    cov_matrix = instance.coverage_matrix

    uncovered_mask = np.ones(n, dtype=bool)
    selected_facilities: List[int] = []
    trace: List[Dict[str, Any]] = []
    iteration = 0
    states_evaluated = 0

    while np.any(uncovered_mask):
        iteration += 1
        best_fac = -1
        best_ratio = float("inf")
        best_new_count = 0

        # Evaluate candidate facilities
        for i in range(m):
            if i in selected_facilities:
                continue

            states_evaluated += 1
            # Newly covered elements = covered by i AND currently uncovered
            newly_covered = np.sum(cov_matrix[i] & uncovered_mask)
            if newly_covered > 0:
                ratio = costs[i] / float(newly_covered)
                # Minimize marginal cost ratio; tie-break by higher coverage then lower cost
                if ratio < best_ratio - 1e-9:
                    best_ratio = ratio
                    best_fac = i
                    best_new_count = int(newly_covered)
                elif abs(ratio - best_ratio) <= 1e-9:
                    if newly_covered > best_new_count or (newly_covered == best_new_count and costs[i] < costs[best_fac]):
                        best_ratio = ratio
                        best_fac = i
                        best_new_count = int(newly_covered)

        if best_fac == -1:
            # Fallback in case instance was somehow disconnected
            break

        selected_facilities.append(best_fac)
        uncovered_mask = uncovered_mask & ~cov_matrix[best_fac]
        cumulative_covered = int(n - np.sum(uncovered_mask))
        cum_pct = float(np.round((cumulative_covered / n) * 100.0, 1))

        trace.append({
            "iteration": iteration,
            "selected_facility": best_fac,
            "facility_cost": float(np.round(costs[best_fac], 3)),
            "newly_covered_count": best_new_count,
            "marginal_cost_per_zone": float(np.round(best_ratio, 4)),
            "cumulative_coverage_pct": cum_pct
        })

    runtime_ms = (time.perf_counter() - start_time) * 1000.0
    total_cost = float(np.round(sum(costs[i] for i in selected_facilities), 4))

    # Determine covered zones
    if selected_facilities:
        covered = np.any(cov_matrix[selected_facilities, :], axis=0)
        covered_zones = set(np.where(covered)[0])
    else:
        covered_zones = set()

    is_feasible = (len(covered_zones) == n)

    return SolverResult(
        solver_name="Chvátal's Greedy Approximation",
        selected_facilities=selected_facilities,
        total_cost=total_cost,
        covered_zones=covered_zones,
        is_feasible=is_feasible,
        runtime_ms=float(np.round(runtime_ms, 3)),
        states_evaluated=states_evaluated,
        trace=trace,
        extra_info={"iterations": iteration, "harmonic_bound_hn": instance.harmonic_bound_hn}
    )


def solve_greedy_plus_pruning(
    instance: ProblemInstance
) -> SolverResult:
    """
    Greedy + Redundancy Pruning & 1-Swap Local Improvement.
    
    1. Runs Chvátal's Greedy algorithm.
    2. Redundancy Pruning: Iterates through selected facilities in descending cost order.
       If removing facility f leaves all demand zones covered, f is safely discarded.
    3. 1-Swap Local Search: Tests swapping an included facility f with an unselected facility g
       if c_g < c_f and feasibility is maintained.
    """
    start_time = time.perf_counter()
    n = instance.n_zones
    m = instance.n_facilities
    costs = instance.costs
    cov_matrix = instance.coverage_matrix

    # Step 1: Base Greedy
    greedy_res = solve_greedy_set_cover(instance)
    selected = list(greedy_res.selected_facilities)
    states_evaluated = greedy_res.states_evaluated

    # Step 2: Redundancy Pruning (descending cost order)
    pruned_facilities: List[int] = []
    # Sort selected descending by cost so we attempt to eliminate expensive redundant facilities first
    for fac in sorted(selected, key=lambda x: -costs[x]):
        if len(selected) <= 1:
            break
        remaining = [f for f in selected if f != fac]
        states_evaluated += 1
        if np.all(np.any(cov_matrix[remaining, :], axis=0)):
            selected = remaining
            pruned_facilities.append(fac)

    # Step 3: 1-Swap Local Search Improvement
    swaps_made: List[Tuple[int, int]] = []
    improved = True
    max_passes = 3
    pass_cnt = 0

    while improved and pass_cnt < max_passes:
        improved = False
        pass_cnt += 1
        unselected = [i for i in range(m) if i not in selected]

        for u in list(selected):
            for v in unselected:
                states_evaluated += 1
                if costs[v] < costs[u]:
                    # Candidate replacement
                    test_sol = [f for f in selected if f != u] + [v]
                    if np.all(np.any(cov_matrix[test_sol, :], axis=0)):
                        selected = test_sol
                        swaps_made.append((u, v))
                        improved = True
                        break
            if improved:
                break

    # Secondary quick redundancy pass after swaps
    for fac in sorted(selected, key=lambda x: -costs[x]):
        if len(selected) <= 1:
            break
        remaining = [f for f in selected if f != fac]
        if np.all(np.any(cov_matrix[remaining, :], axis=0)):
            selected = remaining
            pruned_facilities.append(fac)

    runtime_ms = (time.perf_counter() - start_time) * 1000.0
    total_cost = float(np.round(sum(costs[i] for i in selected), 4))

    covered = np.any(cov_matrix[selected, :], axis=0) if selected else np.zeros(n, dtype=bool)
    covered_zones = set(np.where(covered)[0])
    is_feasible = (len(covered_zones) == n)

    return SolverResult(
        solver_name="Greedy + Pruning & Local Search",
        selected_facilities=sorted(selected),
        total_cost=total_cost,
        covered_zones=covered_zones,
        is_feasible=is_feasible,
        runtime_ms=float(np.round(runtime_ms, 3)),
        states_evaluated=states_evaluated,
        trace=greedy_res.trace,
        extra_info={
            "initial_greedy_cost": greedy_res.total_cost,
            "cost_reduction": float(np.round(greedy_res.total_cost - total_cost, 4)),
            "pruned_facilities": pruned_facilities,
            "swaps_made": swaps_made
        }
    )

