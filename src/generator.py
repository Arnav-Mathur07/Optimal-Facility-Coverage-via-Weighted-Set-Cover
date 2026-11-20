"""
Synthetic and adversarial instance generators for the Weighted Set Cover facility problem.
"""

from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np

from .models import ProblemInstance


def _compute_coverage_matrix(
    facilities: np.ndarray,
    radii: np.ndarray,
    zones: np.ndarray
) -> np.ndarray:
    """
    Compute m x n boolean coverage matrix where coverage_matrix[i, j] = True
    iff Euclidean distance between facility i and zone j <= radius[i].
    """
    # facilities: (m, 2), zones: (n, 2)
    # dist: (m, n)
    diff = facilities[:, np.newaxis, :] - zones[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return dist <= (radii[:, np.newaxis] + 1e-9)


def _ensure_full_coverage(
    zones: np.ndarray,
    facilities: np.ndarray,
    radii: np.ndarray,
    costs: np.ndarray,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ensure every demand zone is covered by at least one candidate facility.
    If any zone is uncovered, expand the nearest facility's radius.
    """
    m = facilities.shape[0]
    n = zones.shape[0]
    cov = _compute_coverage_matrix(facilities, radii, zones)
    uncovered = np.where(~np.any(cov, axis=0))[0]

    if len(uncovered) > 0:
        for z_idx in uncovered:
            z_coord = zones[z_idx]
            dists = np.linalg.norm(facilities - z_coord, axis=1)
            nearest_fac = int(np.argmin(dists))
            needed_radius = float(dists[nearest_fac]) + 1.5
            if needed_radius > radii[nearest_fac]:
                radii[nearest_fac] = needed_radius
                # Slightly increase cost to reflect expanded radius
                costs[nearest_fac] = float(np.round(costs[nearest_fac] + 0.05 * (needed_radius ** 1.3), 2))

        cov = _compute_coverage_matrix(facilities, radii, zones)

    return facilities, radii, costs, cov


def generate_uniform_random(
    n_zones: int = 30,
    n_facilities: int = 12,
    radius_min: float = 20.0,
    radius_max: float = 38.0,
    seed: Optional[int] = None
) -> ProblemInstance:
    """
    Generate a uniform random spatial instance on a [0, 100] x [0, 100] grid.
    Facilities have costs proportional to their coverage area plus stochastic noise.
    """
    rng = np.random.default_rng(seed)
    zones = rng.uniform(5.0, 95.0, size=(n_zones, 2))
    facilities = rng.uniform(5.0, 95.0, size=(n_facilities, 2))
    radii = rng.uniform(radius_min, radius_max, size=(n_facilities,))

    # Cost model: base cost + proportional to area (pi * R^2) + random variation
    area_term = 0.04 * (np.pi * (radii ** 2))
    noise = rng.uniform(2.0, 15.0, size=(n_facilities,))
    costs = np.round(15.0 + area_term + noise, 2)

    facilities, radii, costs, cov = _ensure_full_coverage(zones, facilities, radii, costs, rng)

    return ProblemInstance(
        name=f"UniformRandom_{n_zones}z_{n_facilities}f",
        zones=zones,
        facilities=facilities,
        radii=radii,
        costs=costs,
        coverage_matrix=cov,
        generator_type="Uniform Random Spatial",
        metadata={"seed": seed, "radius_range": (radius_min, radius_max)}
    )


def generate_clustered_urban(
    n_zones: int = 35,
    n_facilities: int = 12,
    n_clusters: int = 3,
    seed: Optional[int] = None
) -> ProblemInstance:
    """
    Generate a clustered urban environment simulating downtown districts & suburbs.
    Features:
    - High-density Gaussian clusters of demand zones.
    - Two facility tiers:
      * Tier 1: Local clinics / small hubs (low cost, small radius, inside clusters).
      * Tier 2: Central regional hospitals / hubs (high cost, large radius, peripheral or central).
    """
    rng = np.random.default_rng(seed)

    # Cluster centroids
    cluster_centers = rng.uniform(20.0, 80.0, size=(n_clusters, 2))

    # Assign zones to clusters with Gaussian jitter
    zones_list: List[np.ndarray] = []
    zones_per_cluster = n_zones // n_clusters
    for i in range(n_clusters):
        count = zones_per_cluster + (1 if i < (n_zones % n_clusters) else 0)
        pts = rng.normal(loc=cluster_centers[i], scale=7.5, size=(count, 2))
        zones_list.append(pts)
    zones = np.clip(np.vstack(zones_list), 2.0, 98.0)

    # Facilities: 65% local clinics, 35% regional hubs
    n_local = max(1, int(0.65 * n_facilities))
    n_regional = n_facilities - n_local

    facilities_list: List[np.ndarray] = []
    radii_list: List[float] = []
    costs_list: List[float] = []

    # Local facilities near cluster centers
    for _ in range(n_local):
        chosen_center = cluster_centers[rng.integers(0, n_clusters)]
        loc = chosen_center + rng.normal(scale=6.0, size=(2,))
        radius = float(rng.uniform(14.0, 24.0))
        cost = float(np.round(rng.uniform(15.0, 35.0), 2))
        facilities_list.append(loc)
        radii_list.append(radius)
        costs_list.append(cost)

    # Regional facilities
    for _ in range(n_regional):
        loc = rng.uniform(20.0, 80.0, size=(2,))
        radius = float(rng.uniform(38.0, 55.0))
        cost = float(np.round(rng.uniform(75.0, 130.0), 2))
        facilities_list.append(loc)
        radii_list.append(radius)
        costs_list.append(cost)

    facilities = np.clip(np.array(facilities_list), 2.0, 98.0)
    radii = np.array(radii_list)
    costs = np.array(costs_list)

    facilities, radii, costs, cov = _ensure_full_coverage(zones, facilities, radii, costs, rng)

    return ProblemInstance(
        name=f"ClusteredUrban_{n_zones}z_{n_facilities}f",
        zones=zones,
        facilities=facilities,
        radii=radii,
        costs=costs,
        coverage_matrix=cov,
        generator_type="Clustered Urban Hotspots",
        metadata={"seed": seed, "n_clusters": n_clusters}
    )


def generate_adversarial_harmonic(
    n_zones: int = 15,
    n_facilities: int = 6,
    seed: Optional[int] = None
) -> ProblemInstance:
    """
    Construct the classic worst-case harmonic Set Cover instance (Lovasz/Chvatal).
    
    Structure:
    - Optimal Facility F_OPT: covers all n demand zones with cost 1.05 (or 1.0 + eps).
    - Deceptive Facilities F_1, ..., F_{m-1}:
      Partitions zones into groups of decreasing size or singletons.
      The cost of each deceptive facility F_j is calibrated so that its marginal cost
      per zone (c_j / |G_j|) is strictly smaller than F_OPT's marginal cost at iteration j.
      Greedy greedily picks all deceptive facilities one by one, incurring total cost ~ H_n,
      while the exact optimum only selects F_OPT at cost 1.05.
      
    This yields an empirical approximation ratio rho approaching the theoretical bound H_n.
    Demand zones and facilities are laid out in an intuitive circular geometric pattern for 2D visualization.
    """
    rng = np.random.default_rng(seed)
    n = max(3, n_zones)
    m = max(3, n_facilities)

    # Place demand zones in a circle centered at (50, 50) with radius 35
    center = np.array([50.0, 50.0])
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    zones = np.column_stack([
        center[0] + 35.0 * np.cos(theta),
        center[1] + 35.0 * np.sin(theta)
    ])

    # Deceptive facilities: partition zones into m - 1 subsets
    n_deceptive = m - 1
    # Distribute group sizes geometrically/harmonically
    # We want groups: sizes roughly proportional to powers of 2 or decreasing
    raw_weights = [2.0 ** (n_deceptive - 1 - i) for i in range(n_deceptive)]
    norm_weights = np.array(raw_weights) / sum(raw_weights)
    group_sizes = np.maximum(1, np.round(norm_weights * n).astype(int))

    # Adjust sum of sizes to equal n exactly
    while sum(group_sizes) > n:
        idx = int(np.argmax(group_sizes))
        group_sizes[idx] -= 1
    while sum(group_sizes) < n:
        idx = int(np.argmin(group_sizes))
        group_sizes[idx] += 1

    # Facility 0: Optimal Hub covering all zones
    # Center at (50, 50), radius 42.0 (covers all zones with radius 35)
    f_opt_cost = 1.05
    facilities_list = [center.copy()]
    radii_list = [42.0]
    costs_list = [f_opt_cost]

    # Assign zones to deceptive groups
    zone_indices = np.arange(n)
    groups: List[np.ndarray] = []
    curr = 0
    for sz in group_sizes:
        groups.append(zone_indices[curr:curr + sz])
        curr += sz

    # Deceptive facilities: each covers its corresponding group
    # We calibrate costs so greedy picks groups in order 0, 1, ..., n_deceptive-1
    remaining_elements = n
    coverage_matrix_list = []
    opt_cov = np.ones(n, dtype=bool)
    coverage_matrix_list.append(opt_cov)

    for j, grp in enumerate(groups):
        grp_size = len(grp)
        # Coordinates: centroid of the covered zones
        grp_pts = zones[grp]
        fac_coord = np.mean(grp_pts, axis=0) if grp_size > 0 else center + rng.uniform(-5, 5, size=(2,))
        if grp_size > 1:
            max_d = float(np.max(np.linalg.norm(grp_pts - fac_coord, axis=1))) + 2.0
        else:
            max_d = 5.0
        radius = max(8.0, max_d)

        # Deceptive cost: we need (c_j / grp_size) < (f_opt_cost / remaining_elements)
        # So c_j < f_opt_cost * (grp_size / remaining_elements)
        # We set c_j = 0.95 * f_opt_cost * (grp_size / remaining_elements)
        cost_j = max(0.05, float(0.95 * f_opt_cost * (grp_size / remaining_elements)))
        remaining_elements -= grp_size

        facilities_list.append(fac_coord)
        radii_list.append(radius)
        costs_list.append(np.round(cost_j, 3))

        # Explicit coverage mask for this group
        mask = np.zeros(n, dtype=bool)
        mask[grp] = True
        coverage_matrix_list.append(mask)

    facilities = np.array(facilities_list)
    radii = np.array(radii_list)
    costs = np.array(costs_list)
    coverage_matrix = np.vstack(coverage_matrix_list)

    return ProblemInstance(
        name=f"AdversarialHarmonic_{n}z_{m}f",
        zones=zones,
        facilities=facilities,
        radii=radii,
        costs=costs,
        coverage_matrix=coverage_matrix,
        generator_type="Adversarial Harmonic Worst-Case",
        metadata={"seed": seed, "k_groups": n_deceptive, "f_opt_cost": f_opt_cost}
    )

