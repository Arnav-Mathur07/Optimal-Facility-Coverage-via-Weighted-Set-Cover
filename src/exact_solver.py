"""
Exact solvers for the Weighted Set Cover problem:
1. Exhaustive Subset Enumeration (Brute-Force Baseline)
2. Branch-and-Bound with Admissible Pruning
"""

from __future__ import annotations
import time
import itertools
from typing import List, Set, Tuple, Optional
import numpy as np

from .models import ProblemInstance, SolverResult


def _build_bitmasks(instance: ProblemInstance) -> Tuple[List[int], int]:
    """
    Represent facility coverage as integer bitmasks for high-performance bitwise ops.
    Target mask has the first n bits set to 1: (1 << n) - 1.
    """
    n = instance.n_zones
    m = instance.n_facilities
    target_mask = (1 << n) - 1

    fac_masks: List[int] = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if instance.coverage_matrix[i, j]:
                mask |= (1 << j)
        fac_masks.append(mask)

    return fac_masks, target_mask


def solve_exact_exhaustive(
    instance: ProblemInstance,
    max_facilities: int = 18
) -> SolverResult:
    """
    Exact global optimum via exhaustive subset enumeration.
    
    Guarantees global minimal cost C_OPT.
    Explores candidate subsets of facilities, tracking states evaluated and execution time.
    Safety guard enforces max_facilities <= 18 to prevent exponential O(2^m) timeout.
    """
    m = instance.n_facilities
    n = instance.n_zones

    if m > max_facilities:
        raise ValueError(
            f"Number of facilities m={m} exceeds exhaustive safety limit (max={max_facilities}). "
            f"Use Branch-and-Bound or the Greedy approximation algorithm for large instances."
        )

    start_time = time.perf_counter()
    fac_masks, target_mask = _build_bitmasks(instance)
    costs = instance.costs.tolist()

    best_cost = float("inf")
    best_subset: List[int] = []
    states_evaluated = 0

    # Enumerate all non-empty subsets 1 .. 2^m - 1
    # Iterate by subset cardinality 1 .. m for fast early bounds
    for k in range(1, m + 1):
        for combo in itertools.combinations(range(m), k):
            states_evaluated += 1

            # Quick cost check
            combo_cost = sum(costs[i] for i in combo)
            if combo_cost >= best_cost:
                continue

            # Check coverage via bitwise OR
            combined_mask = 0
            for i in combo:
                combined_mask |= fac_masks[i]
                if combined_mask == target_mask:
                    break

            if combined_mask == target_mask:
                best_cost = combo_cost
                best_subset = list(combo)

    runtime_ms = (time.perf_counter() - start_time) * 1000.0

    # Determine covered zones
    if best_subset:
        covered_zones = set()
        for i in best_subset:
            covered_zones.update(instance.get_coverage_set(i))
    else:
        covered_zones = set()

    is_feasible = (len(covered_zones) == n)

    return SolverResult(
        solver_name="Exact Exhaustive (Brute-Force)",
        selected_facilities=sorted(best_subset),
        total_cost=float(np.round(best_cost, 4)) if best_cost < float("inf") else 0.0,
        covered_zones=covered_zones,
        is_feasible=is_feasible,
        runtime_ms=float(np.round(runtime_ms, 3)),
        states_evaluated=states_evaluated,
        trace=[],
        extra_info={"total_subsets": (1 << m) - 1, "cardinality": len(best_subset)}
    )


def solve_exact_bnb(
    instance: ProblemInstance,
    max_facilities: int = 22
) -> SolverResult:
    """
    Exact global optimum via Branch-and-Bound with dual pruning:
    1. Cost-bound pruning: If current_cost >= best_cost, prune branch.
    2. Reachability pruning: If current_mask | suffix_union != target_mask, prune branch.
    3. Feasibility termination: Once coverage == target_mask, prune deeper additions (since costs > 0).
    """
    m = instance.n_facilities
    n = instance.n_zones

    if m > max_facilities:
        raise ValueError(
            f"m={m} exceeds Branch-and-Bound safety limit ({max_facilities})."
        )

    start_time = time.perf_counter()
    fac_masks, target_mask = _build_bitmasks(instance)
    costs = instance.costs.tolist()

    # Precompute suffix unions for reachability pruning
    suffix_unions = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix_unions[i] = suffix_unions[i + 1] | fac_masks[i]

    best_cost = float("inf")
    best_subset: List[int] = []
    states_evaluated = 0

    def branch(depth: int, current_mask: int, current_cost: float, current_subset: List[int]):
        nonlocal best_cost, best_subset, states_evaluated
        states_evaluated += 1

        # Prune 1: Cost bound
        if current_cost >= best_cost:
            return

        # Prune 2: Full coverage reached
        if current_mask == target_mask:
            best_cost = current_cost
            best_subset = list(current_subset)
            return

        # Prune 3: End of tree
        if depth >= m:
            return

        # Prune 4: Reachability - even taking all remaining facilities cannot cover universe
        if (current_mask | suffix_unions[depth]) != target_mask:
            return

        # Branch with facility depth (Included)
        # Only take if it contributes new coverage
        if (current_mask | fac_masks[depth]) != current_mask:
            current_subset.append(depth)
            branch(
                depth + 1,
                current_mask | fac_masks[depth],
                current_cost + costs[depth],
                current_subset
            )
            current_subset.pop()

        # Branch without facility depth (Excluded)
        branch(depth + 1, current_mask, current_cost, current_subset)

    # Initial branch call
    branch(0, 0, 0.0, [])

    runtime_ms = (time.perf_counter() - start_time) * 1000.0

    if best_subset:
        covered_zones = set()
        for i in best_subset:
            covered_zones.update(instance.get_coverage_set(i))
    else:
        covered_zones = set()

    is_feasible = (len(covered_zones) == n)

    return SolverResult(
        solver_name="Exact Branch-and-Bound",
        selected_facilities=sorted(best_subset),
        total_cost=float(np.round(best_cost, 4)) if best_cost < float("inf") else 0.0,
        covered_zones=covered_zones,
        is_feasible=is_feasible,
        runtime_ms=float(np.round(runtime_ms, 3)),
        states_evaluated=states_evaluated,
        trace=[],
        extra_info={"cardinality": len(best_subset)}
    )

