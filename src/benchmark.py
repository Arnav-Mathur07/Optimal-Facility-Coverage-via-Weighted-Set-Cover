"""
Benchmarking engine for empirical approximation ratio analysis and runtime scaling.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Callable
import time
import numpy as np
import pandas as pd

from .models import ProblemInstance, BenchmarkRecord
from .generator import (
    generate_uniform_random,
    generate_clustered_urban,
    generate_adversarial_harmonic
)
from .exact_solver import solve_exact_exhaustive, solve_exact_bnb
from .approx_solver import solve_greedy_set_cover, solve_greedy_plus_pruning


def run_single_benchmark(
    instance: ProblemInstance,
    instance_id: int = 1,
    seed: Optional[int] = None,
    use_bnb_for_exact: bool = False
) -> BenchmarkRecord:
    """
    Run exact solver, greedy solver, and refined greedy solver on a single instance,
    computing the empirical approximation ratio and performance metrics.
    """
    # 1. Exact solver
    if use_bnb_for_exact or instance.n_facilities > 18:
        opt_res = solve_exact_bnb(instance)
    else:
        opt_res = solve_exact_exhaustive(instance)

    # 2. Greedy solver
    greedy_res = solve_greedy_set_cover(instance)

    # 3. Refined greedy solver (Pruning + Local Search)
    refined_res = solve_greedy_plus_pruning(instance)

    # Approximation ratios
    c_opt = opt_res.total_cost
    if c_opt > 0:
        emp_ratio = float(np.round(greedy_res.total_cost / c_opt, 4))
        ref_ratio = float(np.round(refined_res.total_cost / c_opt, 4))
    else:
        emp_ratio = 1.0
        ref_ratio = 1.0

    solved_opt = bool(abs(greedy_res.total_cost - c_opt) < 1e-4)
    ref_solved_opt = bool(abs(refined_res.total_cost - c_opt) < 1e-4)

    return BenchmarkRecord(
        instance_id=instance_id,
        generator_type=instance.generator_type,
        n_zones=instance.n_zones,
        n_facilities=instance.n_facilities,
        opt_cost=opt_res.total_cost,
        greedy_cost=greedy_res.total_cost,
        refined_cost=refined_res.total_cost,
        empirical_ratio=emp_ratio,
        refined_ratio=ref_ratio,
        theoretical_bound_hn=float(np.round(instance.harmonic_bound_hn, 4)),
        opt_runtime_ms=opt_res.runtime_ms,
        greedy_runtime_ms=greedy_res.runtime_ms,
        refined_runtime_ms=refined_res.runtime_ms,
        opt_states=opt_res.states_evaluated,
        greedy_states=greedy_res.states_evaluated,
        solved_optimally=solved_opt,
        refined_solved_optimally=ref_solved_opt,
        seed=seed
    )


def run_benchmark_suite(
    k_instances: int = 20,
    generator_type: str = "Uniform Random Spatial",
    n_zones: int = 25,
    n_facilities: int = 12,
    base_seed: int = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run benchmark across K instances, computing empirical ratios and summary statistics.
    """
    records: List[BenchmarkRecord] = []

    for t in range(k_instances):
        seed = base_seed + t
        if generator_type == "Uniform Random Spatial":
            inst = generate_uniform_random(n_zones=n_zones, n_facilities=n_facilities, seed=seed)
        elif generator_type == "Clustered Urban Hotspots":
            inst = generate_clustered_urban(n_zones=n_zones, n_facilities=n_facilities, seed=seed)
        elif generator_type == "Adversarial Harmonic Worst-Case":
            inst = generate_adversarial_harmonic(n_zones=n_zones, n_facilities=n_facilities, seed=seed)
        else:
            inst = generate_uniform_random(n_zones=n_zones, n_facilities=n_facilities, seed=seed)

        rec = run_single_benchmark(inst, instance_id=t + 1, seed=seed)
        records.append(rec)

        if progress_callback:
            progress_callback(t + 1, k_instances)

    df = pd.DataFrame([vars(r) for r in records])

    # Summary statistics
    mean_rho = float(df["empirical_ratio"].mean())
    median_rho = float(df["empirical_ratio"].median())
    max_rho = float(df["empirical_ratio"].max())
    mean_ref_rho = float(df["refined_ratio"].mean())
    optimal_pct = float((df["solved_optimally"].sum() / len(df)) * 100.0)
    ref_optimal_pct = float((df["refined_solved_optimally"].sum() / len(df)) * 100.0)
    mean_speedup = float((df["opt_runtime_ms"] / np.maximum(df["greedy_runtime_ms"], 0.001)).mean())
    mean_hn = float(df["theoretical_bound_hn"].mean())

    summary = {
        "mean_ratio": round(mean_rho, 4),
        "median_ratio": round(median_rho, 4),
        "max_ratio": round(max_rho, 4),
        "mean_refined_ratio": round(mean_ref_rho, 4),
        "optimal_hit_rate_pct": round(optimal_pct, 1),
        "refined_optimal_hit_rate_pct": round(ref_optimal_pct, 1),
        "speedup_factor": round(mean_speedup, 1),
        "mean_theoretical_hn": round(mean_hn, 4),
        "k_instances": k_instances
    }

    return df, summary


def run_runtime_scaling_benchmark(
    m_range: List[int] = list(range(6, 17)),
    fixed_n_zones: int = 30,
    seed: int = 42
) -> pd.DataFrame:
    """
    Benchmark wall-clock runtime (ms) and states evaluated as facility count m scales,
    comparing Exhaustive Brute-Force vs Branch-and-Bound vs Greedy.
    """
    rows = []

    for m in m_range:
        inst = generate_uniform_random(n_zones=fixed_n_zones, n_facilities=m, seed=seed + m)

        # 1. Greedy
        g_res = solve_greedy_set_cover(inst)

        # 2. Branch-and-Bound
        bnb_res = solve_exact_bnb(inst)

        # 3. Exhaustive (if m <= 16)
        if m <= 16:
            ex_res = solve_exact_exhaustive(inst)
            ex_time = ex_res.runtime_ms
            ex_states = ex_res.states_evaluated
            ex_cost = ex_res.total_cost
        else:
            ex_time = np.nan
            ex_states = (1 << m) - 1
            ex_cost = bnb_res.total_cost

        rows.append({
            "m_facilities": m,
            "n_zones": fixed_n_zones,
            "greedy_time_ms": g_res.runtime_ms,
            "bnb_time_ms": bnb_res.runtime_ms,
            "exhaustive_time_ms": ex_time,
            "greedy_states": g_res.states_evaluated,
            "bnb_states": bnb_res.states_evaluated,
            "exhaustive_states": ex_states,
            "greedy_cost": g_res.total_cost,
            "exact_opt_cost": bnb_res.total_cost,
            "ratio": round(g_res.total_cost / bnb_res.total_cost, 4) if bnb_res.total_cost > 0 else 1.0
        })

    return pd.DataFrame(rows)

