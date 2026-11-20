"""
Data models and representations for the Weighted Set Cover facility coverage problem.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Optional
import numpy as np


@dataclass
class ProblemInstance:
    """
    Representation of a 2D Weighted Set Cover problem instance for facility coverage.
    
    Universe U: n demand zones {z_0, ..., z_{n-1}} in [0, 100] x [0, 100].
    Collection S: m candidate facility sites {F_0, ..., F_{m-1}}, each with
    coordinates (x_i, y_i), radius R_i, installation cost c_i, and coverage subset S_i.
    """
    name: str
    zones: np.ndarray          # shape (n, 2): (x, y) coordinates
    facilities: np.ndarray     # shape (m, 2): (x, y) coordinates
    radii: np.ndarray          # shape (m,): coverage radius per facility
    costs: np.ndarray          # shape (m,): positive installation cost per facility
    coverage_matrix: np.ndarray # shape (m, n) bool: coverage_matrix[i, j] is True if facility i covers zone j
    generator_type: str = "Uniform Random Spatial"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_zones(self) -> int:
        return self.zones.shape[0]

    @property
    def n_facilities(self) -> int:
        return self.facilities.shape[0]

    @property
    def harmonic_bound_hn(self) -> float:
        """Harmonic number H_n = sum_{k=1}^n 1/k."""
        if self.n_zones == 0:
            return 1.0
        return float(np.sum(1.0 / np.arange(1, self.n_zones + 1)))

    @property
    def is_coverable(self) -> bool:
        """True if every zone is covered by at least one facility."""
        if self.n_zones == 0:
            return True
        return bool(np.all(np.any(self.coverage_matrix, axis=0)))

    def get_coverage_set(self, facility_idx: int) -> Set[int]:
        """Return the set of zone indices covered by facility_idx."""
        return set(np.where(self.coverage_matrix[facility_idx])[0])

    def get_uncovered_zones(self, selected_facilities: List[int]) -> Set[int]:
        """Return set of zone indices not covered by any of selected_facilities."""
        if not selected_facilities:
            return set(range(self.n_zones))
        covered = np.any(self.coverage_matrix[selected_facilities, :], axis=0)
        return set(np.where(~covered)[0])


@dataclass
class SolverResult:
    """
    Output structure for exact or heuristic Set Cover solvers.
    """
    solver_name: str
    selected_facilities: List[int]
    total_cost: float
    covered_zones: Set[int]
    is_feasible: bool
    runtime_ms: float
    states_evaluated: int
    trace: List[Dict[str, Any]] = field(default_factory=list)
    extra_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def coverage_pct(self) -> float:
        return len(self.covered_zones) / max(1, len(self.covered_zones)) * 100.0


@dataclass
class BenchmarkRecord:
    """
    Single benchmark record for empirical approximation analysis.
    """
    instance_id: int
    generator_type: str
    n_zones: int
    n_facilities: int
    opt_cost: float
    greedy_cost: float
    refined_cost: float
    empirical_ratio: float           # greedy_cost / opt_cost
    refined_ratio: float             # refined_cost / opt_cost
    theoretical_bound_hn: float      # H_n
    opt_runtime_ms: float
    greedy_runtime_ms: float
    refined_runtime_ms: float
    opt_states: int
    greedy_states: int
    solved_optimally: bool
    refined_solved_optimally: bool
    seed: Optional[int] = None

