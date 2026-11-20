"""
Automated unit and invariant verification tests for Weighted Set Cover solvers:
- 100% feasibility & universe coverage across all generators and solvers
- Strict optimality invariant: cost(greedy) >= cost(exact_opt) (rho >= 1.0)
- Branch-and-Bound exactness: cost(BnB) == cost(Exhaustive) and states(BnB) <= states(Exhaustive)
- Worst-case harmonic bound: rho > 1.3 and rho <= H_n + 1e-6 on adversarial instances
"""

import pytest
import numpy as np

from src.generator import (
    generate_uniform_random,
    generate_clustered_urban,
    generate_adversarial_harmonic
)
from src.exact_solver import solve_exact_exhaustive, solve_exact_bnb
from src.approx_solver import solve_greedy_set_cover, solve_greedy_plus_pruning


def test_feasibility_guarantee_uniform():
    """Verify that every generated uniform instance is 100% coverable and all solvers yield feasible solutions."""
    for seed in [101, 202, 303]:
        inst = generate_uniform_random(n_zones=25, n_facilities=10, seed=seed)
        assert inst.is_coverable, f"Instance with seed {seed} must be 100% coverable"

        ex_res = solve_exact_exhaustive(inst)
        bnb_res = solve_exact_bnb(inst)
        gr_res = solve_greedy_set_cover(inst)
        ref_res = solve_greedy_plus_pruning(inst)

        assert ex_res.is_feasible, "Exhaustive solver must achieve 100% coverage"
        assert bnb_res.is_feasible, "Branch-and-Bound solver must achieve 100% coverage"
        assert gr_res.is_feasible, "Greedy solver must achieve 100% coverage"
        assert ref_res.is_feasible, "Refined greedy solver must achieve 100% coverage"

        assert len(ex_res.covered_zones) == inst.n_zones
        assert len(gr_res.covered_zones) == inst.n_zones


def test_feasibility_guarantee_clustered():
    """Verify feasibility across clustered urban instances with multiple facility tiers."""
    for seed in [42, 77, 99]:
        inst = generate_clustered_urban(n_zones=30, n_facilities=11, n_clusters=3, seed=seed)
        assert inst.is_coverable
        gr_res = solve_greedy_set_cover(inst)
        assert gr_res.is_feasible
        assert len(gr_res.covered_zones) == inst.n_zones


def test_branch_and_bound_matches_exhaustive():
    """Verify BnB yields identical optimal cost to exhaustive search while evaluating fewer or equal states."""
    for seed in range(5):
        inst = generate_uniform_random(n_zones=20, n_facilities=10, seed=seed + 10)
        ex_res = solve_exact_exhaustive(inst)
        bnb_res = solve_exact_bnb(inst)

        assert abs(ex_res.total_cost - bnb_res.total_cost) < 1e-4, (
            f"BnB cost ({bnb_res.total_cost}) must equal Exhaustive cost ({ex_res.total_cost})"
        )
        assert bnb_res.states_evaluated <= ex_res.states_evaluated, (
            f"BnB states ({bnb_res.states_evaluated}) must be <= Exhaustive states ({ex_res.states_evaluated})"
        )


def test_strict_optimality_invariant_50_instances():
    """
    Verify the strict mathematical invariant:
    cost(greedy) >= cost(exact_opt) - 1e-9 across 50 randomized instances.
    Zero optimality violations allowed.
    """
    violations = 0
    tested = 0

    for seed in range(50):
        # Alternate generators and sizes
        if seed % 2 == 0:
            inst = generate_uniform_random(n_zones=20 + (seed % 10), n_facilities=8 + (seed % 4), seed=seed)
        else:
            inst = generate_clustered_urban(n_zones=20 + (seed % 10), n_facilities=8 + (seed % 4), seed=seed)

        opt_res = solve_exact_bnb(inst)
        gr_res = solve_greedy_set_cover(inst)
        ref_res = solve_greedy_plus_pruning(inst)

        tested += 1

        # Check Greedy vs OPT
        if gr_res.total_cost < opt_res.total_cost - 1e-4:
            violations += 1

        # Check Refined vs OPT
        if ref_res.total_cost < opt_res.total_cost - 1e-4:
            violations += 1

        # Refined cost should always be <= greedy cost
        assert ref_res.total_cost <= gr_res.total_cost + 1e-4

    assert violations == 0, f"Found {violations} violations of the optimality invariant out of {tested} tests"


def test_adversarial_harmonic_worst_case():
    """
    Verify the adversarial harmonic instance:
    1. Approximation ratio rho > 1.3
    2. rho <= H_n + 1e-6 (strictly adheres to the theoretical upper bound)
    """
    inst = generate_adversarial_harmonic(n_zones=15, n_facilities=6, seed=42)
    assert inst.is_coverable

    opt_res = solve_exact_bnb(inst)
    gr_res = solve_greedy_set_cover(inst)

    rho = gr_res.total_cost / opt_res.total_cost
    h_n = inst.harmonic_bound_hn

    assert rho > 1.3, f"Adversarial instance empirical ratio {rho:.4f} must exceed 1.3"
    assert rho <= h_n + 1e-6, f"Empirical ratio {rho:.4f} must not exceed theoretical bound H_n={h_n:.4f}"


def test_greedy_trace_integrity():
    """Verify the trace contains valid monotonic cumulative coverage up to 100%."""
    inst = generate_uniform_random(n_zones=25, n_facilities=10, seed=7)
    res = solve_greedy_set_cover(inst)

    assert len(res.trace) > 0
    cum_pcts = [step["cumulative_coverage_pct"] for step in res.trace]

    # Monotonically non-decreasing coverage
    for i in range(1, len(cum_pcts)):
        assert cum_pcts[i] >= cum_pcts[i - 1]

    # Reaches 100%
    assert cum_pcts[-1] == 100.0

