"""
Design & Analysis of Algorithms (DAA) PBL Interactive Dashboard
NP-Hard Minimization Problem: Weighted Set Cover for Optimal Facility Coverage

Publication-grade interactive dashboard comparing:
1. Exact Global Optimum (Exhaustive Combinatorial Enumeration & Branch-and-Bound)
2. Chvátal's Greedy Approximation Algorithm (H_n Harmonic Ratio)
3. Greedy + Redundancy Pruning & 1-Swap Local Search Refinement
"""

from __future__ import annotations
import time
from typing import List, Set, Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.models import ProblemInstance, SolverResult, BenchmarkRecord
from src.generator import (
    generate_uniform_random,
    generate_clustered_urban,
    generate_adversarial_harmonic
)
from src.exact_solver import solve_exact_exhaustive, solve_exact_bnb
from src.approx_solver import solve_greedy_set_cover, solve_greedy_plus_pruning
from src.benchmark import (
    run_single_benchmark,
    run_benchmark_suite,
    run_runtime_scaling_benchmark
)

# ---------------------------------------------------------
# Streamlit Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Weighted Set Cover - Optimal Facility Coverage",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Academic Theme CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px 20px;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 12px;
    }
    .metric-card-opt { border-left-color: #2E7D32; }
    .metric-card-greedy { border-left-color: #EF6C00; }
    .metric-card-ref { border-left-color: #00897B; }
    .metric-card-ratio { border-left-color: #8E24AA; }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #5f6368;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #202124;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #70757a;
        margin-top: 2px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Plotly Spatial Visualization Helper
# ---------------------------------------------------------
def create_spatial_map(
    instance: ProblemInstance,
    selected_facility_indices: List[int],
    title: str,
    accent_color: str = "#1E88E5",
    accent_name: str = "Selected"
) -> go.Figure:
    """
    Render 2D spatial map of demand zones, candidate facilities, and coverage radii.
    """
    fig = go.Figure()

    selected_set = set(selected_facility_indices)
    theta = np.linspace(0, 2 * np.pi, 60)

    # 1. Facility coverage perimeter circles
    # Unselected facilities (dimmed dotted outline)
    for i in range(instance.n_facilities):
        if i not in selected_set:
            cx, cy = instance.facilities[i]
            r = instance.radii[i]
            circle_x = cx + r * np.cos(theta)
            circle_y = cy + r * np.sin(theta)
            fig.add_trace(go.Scatter(
                x=circle_x,
                y=circle_y,
                mode="lines",
                line=dict(color="rgba(180, 180, 180, 0.35)", width=1, dash="dot"),
                hoverinfo="skip",
                showlegend=False
            ))

    # Selected facilities (prominent circle with soft fill)
    for i in selected_facility_indices:
        cx, cy = instance.facilities[i]
        r = instance.radii[i]
        circle_x = cx + r * np.cos(theta)
        circle_y = cy + r * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            fill="toself",
            fillcolor=f"rgba({int(accent_color[1:3], 16)}, {int(accent_color[3:5], 16)}, {int(accent_color[5:7], 16)}, 0.12)",
            line=dict(color=accent_color, width=2),
            name=f"Coverage F_{i}",
            hoverinfo="skip",
            showlegend=False
        ))

    # 2. Demand Zones (all covered in feasible solution)
    fig.add_trace(go.Scatter(
        x=instance.zones[:, 0],
        y=instance.zones[:, 1],
        mode="markers",
        marker=dict(
            size=7,
            color="#D84315",
            symbol="circle",
            line=dict(width=1, color="#BF360C")
        ),
        name=f"Demand Zones (n={instance.n_zones})",
        hovertemplate="<b>Demand Zone z_%{pointNumber}</b><br>Coordinates: (%{x:.1f}, %{y:.1f})<extra></extra>"
    ))

    # 3. Candidate Facilities (Unselected)
    unselected_indices = [i for i in range(instance.n_facilities) if i not in selected_set]
    if unselected_indices:
        fig.add_trace(go.Scatter(
            x=instance.facilities[unselected_indices, 0],
            y=instance.facilities[unselected_indices, 1],
            mode="markers+text",
            marker=dict(
                size=11,
                color="#9E9E9E",
                symbol="square-open",
                line=dict(width=2, color="#757575")
            ),
            text=[f"F{i}" for i in unselected_indices],
            textposition="top center",
            name="Unselected Facilities",
            customdata=[[i, instance.costs[i], instance.radii[i]] for i in unselected_indices],
            hovertemplate="<b>Facility F_%{customdata[0]} (Unselected)</b><br>Cost: $%{customdata[1]:.2f}<br>Radius: %{customdata[2]:.1f}<br>Coord: (%{x:.1f}, %{y:.1f})<extra></extra>"
        ))

    # 4. Selected Facilities
    if selected_facility_indices:
        fig.add_trace(go.Scatter(
            x=instance.facilities[selected_facility_indices, 0],
            y=instance.facilities[selected_facility_indices, 1],
            mode="markers+text",
            marker=dict(
                size=14,
                color=accent_color,
                symbol="diamond",
                line=dict(width=2, color="#212121")
            ),
            text=[f"F{i}" for i in selected_facility_indices],
            textposition="top center",
            name=f"{accent_name} ({len(selected_facility_indices)} Sites)",
            customdata=[[i, instance.costs[i], instance.radii[i]] for i in selected_facility_indices],
            hovertemplate=f"<b>Facility F_%{{customdata[0]}} ({accent_name})</b><br>Cost: $%{{customdata[1]:.2f}}<br>Radius: %{{customdata[2]:.1f}}<br>Coord: (%{{x:.1f}}, %{{y:.1f}})<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#202124")),
        xaxis=dict(range=[-5, 105], showgrid=True, gridcolor="#ECEFF1", zeroline=False, title="X Coordinate"),
        yaxis=dict(range=[-5, 105], showgrid=True, gridcolor="#ECEFF1", zeroline=False, title="Y Coordinate", scaleanchor="x", scaleratio=1),
        margin=dict(l=30, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=480
    )

    return fig


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.title("⚙️ Problem & Solver Controls")
st.sidebar.markdown("Configure spatial distribution, candidate facilities, and solver constraints.")

gen_type = st.sidebar.selectbox(
    "Spatial Generator Mode",
    options=["Uniform Random Spatial", "Clustered Urban Hotspots", "Adversarial Harmonic Worst-Case"],
    index=0
)

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    m_facilities = st.slider("Facilities (m)", min_value=6, max_value=16, value=12, step=1,
                             help="Candidate facilities available. Kept <= 16 for instantaneous exact combinatorial search.")
with col_s2:
    n_zones = st.slider("Demand Zones (n)", min_value=10, max_value=60, value=30, step=5,
                        help="Discrete consumer/demand locations requiring coverage.")

if gen_type != "Adversarial Harmonic Worst-Case":
    radius_val = st.sidebar.slider("Nominal Coverage Radius (R)", min_value=15.0, max_value=45.0, value=28.0, step=1.0)
else:
    radius_val = 35.0
    st.sidebar.info("💡 Adversarial mode uses calibrated geometric subsets to trigger worst-case harmonic behavior.")

seed_val = st.sidebar.number_input("Random Seed", min_value=0, max_value=999999, value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Exact Baseline Engine")
exact_solver_choice = st.sidebar.radio(
    "Exact Solver Mode",
    options=["Exact Exhaustive (Brute-Force Combinatorial)", "Branch-and-Bound (Cost + Feasibility Pruning)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("Benchmark Settings")
batch_k = st.sidebar.slider("Benchmark Batch Size (K)", min_value=10, max_value=50, value=20, step=5)


# ---------------------------------------------------------
# Generate Active Instance & Solve
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_instance(g_type: str, n: int, m: int, r_nom: float, seed: int) -> ProblemInstance:
    if g_type == "Uniform Random Spatial":
        return generate_uniform_random(n_zones=n, n_facilities=m, radius_min=r_nom * 0.75, radius_max=r_nom * 1.25, seed=seed)
    elif g_type == "Clustered Urban Hotspots":
        return generate_clustered_urban(n_zones=n, n_facilities=m, seed=seed)
    else:
        return generate_adversarial_harmonic(n_zones=n, n_facilities=m, seed=seed)


active_instance = get_cached_instance(gen_type, n_zones, m_facilities, radius_val, seed_val)

# Execute Solvers
use_bnb = (exact_solver_choice == "Branch-and-Bound (Cost + Feasibility Pruning)")
if use_bnb:
    exact_res = solve_exact_bnb(active_instance)
else:
    exact_res = solve_exact_exhaustive(active_instance)

greedy_res = solve_greedy_set_cover(active_instance)
refined_res = solve_greedy_plus_pruning(active_instance)

# Performance Metrics
c_opt = exact_res.total_cost
c_greedy = greedy_res.total_cost
c_refined = refined_res.total_cost

emp_ratio = c_greedy / c_opt if c_opt > 0 else 1.0
ref_ratio = c_refined / c_opt if c_opt > 0 else 1.0
h_n = active_instance.harmonic_bound_hn
speedup = exact_res.runtime_ms / max(0.001, greedy_res.runtime_ms)


# ---------------------------------------------------------
# Main UI Dashboard Header
# ---------------------------------------------------------
st.title("📡 Optimal Facility Coverage via Weighted Set Cover")
st.markdown("""
**DAA Project-Based Learning Computational Testbed** &mdash; Rigorous empirical comparison of 
**Chvátal's Greedy Approximation Algorithm** ($\mathcal{O}(m \cdot n)$) vs. **Exact Global Optimality** ($\mathcal{O}(2^m \cdot n)$).
""")

# Main Tabs
tab1, tab2, tab3 = st.tabs([
    "🗺️ Tab 1: Single-Instance Spatial & Decision Explorer",
    "📊 Tab 2: Multi-Instance Empirical Approximation Benchmark",
    "📖 Tab 3: Algorithm Theory & Complexity Analysis"
])


# =========================================================
# TAB 1: Single-Instance Spatial & Decision Explorer
# =========================================================
with tab1:
    # Top KPI Metrics Row
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)

    with kpi_col1:
        st.markdown(f"""
        <div class="metric-card metric-card-opt">
            <div class="metric-title">Optimal Cost (OPT)</div>
            <div class="metric-value">${c_opt:.2f}</div>
            <div class="metric-sub">{len(exact_res.selected_facilities)} facilities selected</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        st.markdown(f"""
        <div class="metric-card metric-card-greedy">
            <div class="metric-title">Greedy Cost</div>
            <div class="metric-value">${c_greedy:.2f}</div>
            <div class="metric-sub">{len(greedy_res.selected_facilities)} facilities selected</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div class="metric-card metric-card-ref">
            <div class="metric-title">Refined Greedy (1-Opt)</div>
            <div class="metric-value">${c_refined:.2f}</div>
            <div class="metric-sub">{len(refined_res.selected_facilities)} facilities selected</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        st.markdown(f"""
        <div class="metric-card metric-card-ratio">
            <div class="metric-title">Empirical Ratio (ρ)</div>
            <div class="metric-value">{emp_ratio:.3f}</div>
            <div class="metric-sub">Refined ρ: {ref_ratio:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Harmonic Bound (H_n)</div>
            <div class="metric-value">{h_n:.3f}</div>
            <div class="metric-sub">Theoretical ceiling</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Greedy Speedup</div>
            <div class="metric-value">{speedup:.1f}×</div>
            <div class="metric-sub">{greedy_res.runtime_ms:.2f}ms vs {exact_res.runtime_ms:.1f}ms</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Side-by-Side 2D Spatial Coverage Maps
    map_col1, map_col2 = st.columns(2)

    with map_col1:
        fig_opt = create_spatial_map(
            active_instance,
            exact_res.selected_facilities,
            f"Exact Global Optimum (Cost: ${c_opt:.2f})",
            accent_color="#2E7D32",
            accent_name="OPT Selected"
        )
        st.plotly_chart(fig_opt, use_container_width=True)

    with map_col2:
        fig_greedy = create_spatial_map(
            active_instance,
            greedy_res.selected_facilities,
            f"Chvátal's Greedy Approximation (Cost: ${c_greedy:.2f}, ρ = {emp_ratio:.3f})",
            accent_color="#EF6C00",
            accent_name="Greedy Selected"
        )
        st.plotly_chart(fig_greedy, use_container_width=True)

    st.markdown("---")

    # Step-by-Step Greedy Trace & Dynamic Convergence
    st.subheader("📋 Step-by-Step Greedy Decision Trace")
    st.markdown(
        "At each iteration, Chvátal's algorithm calculates the marginal cost-effectiveness "
        r"$\alpha_i = \frac{c_i}{|S_i \cap U_{\text{rem}}|}$ and greedily commits the facility with minimal $\alpha_i$."
    )

    trace_data = []
    for step in greedy_res.trace:
        trace_data.append({
            "Iteration": step["iteration"],
            "Selected Facility": f"F_{step['selected_facility']}",
            "Facility Cost ($)": f"${step['facility_cost']:.2f}",
            "Newly Covered Zones (|ΔU|)": step["newly_covered_count"],
            "Marginal Cost per Zone ($/zone)": f"${step['marginal_cost_per_zone']:.3f}",
            "Cumulative Coverage (%)": f"{step['cumulative_coverage_pct']:.1f}%"
        })

    trace_df = pd.DataFrame(trace_data)

    trace_col1, trace_col2 = st.columns([3, 2])

    with trace_col1:
        st.dataframe(trace_df, use_container_width=True, hide_index=True)

    with trace_col2:
        # Dual-axis trace chart
        iter_nums = [s["iteration"] for s in greedy_res.trace]
        cov_pcts = [s["cumulative_coverage_pct"] for s in greedy_res.trace]
        marg_costs = [s["marginal_cost_per_zone"] for s in greedy_res.trace]

        fig_trace = go.Figure()
        fig_trace.add_trace(go.Bar(
            x=iter_nums,
            y=marg_costs,
            name="Marginal Cost ($/zone)",
            marker_color="#FFA726",
            yaxis="y1"
        ))
        fig_trace.add_trace(go.Scatter(
            x=iter_nums,
            y=cov_pcts,
            name="Cumulative Coverage (%)",
            mode="lines+markers",
            line=dict(color="#1E88E5", width=3),
            yaxis="y2"
        ))

        fig_trace.update_layout(
            title="Greedy Progress: Marginal Cost vs Coverage",
            xaxis=dict(title="Iteration", dtick=1),
            yaxis=dict(title="Marginal Cost ($)", side="left"),
            yaxis2=dict(title="Coverage (%)", overlaying="y", side="right", range=[0, 105]),
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            height=280
        )
        st.plotly_chart(fig_trace, use_container_width=True)

    # Local Search refinement summary
    if refined_res.extra_info.get("cost_reduction", 0.0) > 0:
        st.success(
            f"✨ **1-Opt Post-Processing Improvement:** Redundancy pruning and local search reduced cost by "
            f"**${refined_res.extra_info['cost_reduction']:.2f}** (Pruned: {refined_res.extra_info['pruned_facilities']}, "
            f"Swaps: {refined_res.extra_info['swaps_made']}). Refined ratio: **ρ = {ref_ratio:.3f}**."
        )


# =========================================================
# TAB 2: Multi-Instance Empirical Approximation Benchmark
# =========================================================
with tab2:
    st.subheader(f"📊 Empirical Approximation Benchmark Across {batch_k} Test Instances")
    st.markdown("""
    Evaluate approximation robustness across independent synthetic instances. 
    Verify the fundamental empirical invariant: **$1.0 \le \rho \le H_n$**.
    """)

    # Cache benchmark runs in session state
    benchmark_key = f"bench_{gen_type}_{n_zones}_{m_facilities}_{batch_k}_{seed_val}"
    run_benchmark = st.button("🚀 Run Live Benchmark Suite", type="primary")

    if run_benchmark or benchmark_key not in st.session_state:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(curr, total):
            progress_bar.progress(curr / total)
            status_text.text(f"Solving instance {curr}/{total}...")

        with st.spinner("Executing benchmark across test instances..."):
            bench_df, bench_summary = run_benchmark_suite(
                k_instances=batch_k,
                generator_type=gen_type,
                n_zones=n_zones,
                n_facilities=m_facilities,
                base_seed=seed_val,
                progress_callback=update_progress
            )
            st.session_state[benchmark_key] = (bench_df, bench_summary)
            progress_bar.empty()
            status_text.empty()

    bench_df, bench_summary = st.session_state[benchmark_key]

    # Summary Metrics Banner
    b_col1, b_col2, b_col3, b_col4, b_col5, b_col6 = st.columns(6)
    with b_col1:
        st.metric("Mean Ratio (ρ̄)", f"{bench_summary['mean_ratio']:.3f}", f"H_n = {bench_summary['mean_theoretical_hn']:.2f}")
    with b_col2:
        st.metric("Median Ratio", f"{bench_summary['median_ratio']:.3f}")
    with b_col3:
        st.metric("Worst-Case (Max) ρ", f"{bench_summary['max_ratio']:.3f}")
    with b_col4:
        st.metric("Optimal Hit Rate", f"{bench_summary['optimal_hit_rate_pct']:.1f}%", "ρ = 1.00 exactly")
    with b_col5:
        st.metric("Refined Hit Rate", f"{bench_summary['refined_optimal_hit_rate_pct']:.1f}%", "Greedy + 1-Opt")
    with b_col6:
        st.metric("Avg Speedup Factor", f"{bench_summary['speedup_factor']:.1f}×", "Time_OPT / Time_Greedy")

    st.markdown("---")

    # Chart 1: Approximation Ratio across instances
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_ratios = go.Figure()
        fig_ratios.add_trace(go.Scatter(
            x=bench_df["instance_id"],
            y=bench_df["empirical_ratio"],
            mode="lines+markers",
            marker=dict(size=8, color="#EF6C00"),
            line=dict(color="#FFA726", width=2),
            name="Greedy Ratio (ρ)"
        ))
        fig_ratios.add_trace(go.Scatter(
            x=bench_df["instance_id"],
            y=bench_df["refined_ratio"],
            mode="lines+markers",
            marker=dict(size=6, color="#00897B"),
            line=dict(color="#26A69A", width=1.5, dash="dash"),
            name="Refined Ratio (ρ_ref)"
        ))

        # Horizontal references
        fig_ratios.add_hline(y=1.0, line_dash="solid", line_color="#2E7D32", line_width=2,
                             annotation_text="Global Optimum (ρ = 1.0)", annotation_position="bottom right")
        fig_ratios.add_hline(y=bench_summary['mean_theoretical_hn'], line_dash="dash", line_color="#C62828", line_width=1.5,
                             annotation_text=f"Theoretical Bound (H_n = {bench_summary['mean_theoretical_hn']:.2f})",
                             annotation_position="top right")

        fig_ratios.update_layout(
            title="Chart 1: Empirical Approximation Ratio Across Instances",
            xaxis=dict(title="Instance Index"),
            yaxis=dict(title="Approximation Ratio (ρ)", range=[0.9, max(bench_summary['max_ratio'] + 0.3, bench_summary['mean_theoretical_hn'] + 0.2)]),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            margin=dict(l=40, r=40, t=50, b=40),
            height=380
        )
        st.plotly_chart(fig_ratios, use_container_width=True)

    with chart_col2:
        # Chart 2: Cost Comparison
        fig_costs = go.Figure()
        fig_costs.add_trace(go.Bar(
            x=bench_df["instance_id"],
            y=bench_df["opt_cost"],
            name="Exact OPT Cost",
            marker_color="#2E7D32"
        ))
        fig_costs.add_trace(go.Bar(
            x=bench_df["instance_id"],
            y=bench_df["greedy_cost"],
            name="Greedy Cost",
            marker_color="#EF6C00"
        ))
        fig_costs.add_trace(go.Bar(
            x=bench_df["instance_id"],
            y=bench_df["refined_cost"],
            name="Refined Cost",
            marker_color="#00897B"
        ))

        fig_costs.update_layout(
            title="Chart 2: Solution Cost Comparison ($)",
            barmode="group",
            xaxis=dict(title="Instance Index"),
            yaxis=dict(title="Total Installation Cost ($)"),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            margin=dict(l=40, r=40, t=50, b=40),
            height=380
        )
        st.plotly_chart(fig_costs, use_container_width=True)

    st.markdown("---")

    # Chart 3 & Chart 4: Ratio Distribution & Asymptotic Runtime Scaling
    dist_col, scale_col = st.columns(2)

    with dist_col:
        fig_dist = px.histogram(
            bench_df,
            x="empirical_ratio",
            nbins=12,
            marginal="box",
            title="Chart 3: Distribution of Empirical Approximation Ratios",
            labels={"empirical_ratio": "Approximation Ratio (ρ)"},
            color_discrete_sequence=["#EF6C00"]
        )
        fig_dist.add_vline(x=1.0, line_dash="solid", line_color="#2E7D32", annotation_text="OPT (1.0)")
        fig_dist.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=40, r=40, t=50, b=40),
            height=380
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with scale_col:
        # Runtime Scaling analysis button
        st.markdown("**Chart 4: Asymptotic Runtime Scaling ($m = 6 \\dots 15$)**")
        scale_key = "runtime_scaling_data"
        if scale_key not in st.session_state:
            with st.spinner("Measuring asymptotic runtime scaling across facility counts..."):
                st.session_state[scale_key] = run_runtime_scaling_benchmark(
                    m_range=list(range(6, 16)),
                    fixed_n_zones=25,
                    seed=42
                )

        scale_df = st.session_state[scale_key]

        fig_scale = go.Figure()
        fig_scale.add_trace(go.Scatter(
            x=scale_df["m_facilities"],
            y=scale_df["exhaustive_time_ms"],
            mode="lines+markers",
            marker=dict(size=7, color="#C62828"),
            line=dict(color="#E53935", width=2.5),
            name="Exhaustive Brute-Force (O(2^m))"
        ))
        fig_scale.add_trace(go.Scatter(
            x=scale_df["m_facilities"],
            y=scale_df["bnb_time_ms"],
            mode="lines+markers",
            marker=dict(size=7, color="#1565C0"),
            line=dict(color="#1E88E5", width=2.5),
            name="Branch-and-Bound (Pruned Tree)"
        ))
        fig_scale.add_trace(go.Scatter(
            x=scale_df["m_facilities"],
            y=scale_df["greedy_time_ms"],
            mode="lines+markers",
            marker=dict(size=7, color="#2E7D32"),
            line=dict(color="#43A047", width=2.5),
            name="Chvátal's Greedy (O(m·n))"
        ))

        fig_scale.update_layout(
            title="Runtime vs Facility Count (Log Scale)",
            xaxis=dict(title="Candidate Facilities (m)", dtick=1),
            yaxis=dict(title="Wall-Clock Runtime (ms)", type="log"),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            margin=dict(l=40, r=40, t=50, b=40),
            height=380
        )
        st.plotly_chart(fig_scale, use_container_width=True)

    # Export Full Benchmark Table
    st.markdown("---")
    st.subheader("📑 Raw Benchmark Execution Records")
    st.dataframe(bench_df, use_container_width=True)

    csv_data = bench_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Benchmark Results (CSV)",
        data=csv_data,
        file_name=f"set_cover_benchmark_{gen_type}_{batch_k}instances.csv",
        mime="text/csv"
    )


# =========================================================
# TAB 3: Algorithm Theory & Complexity Analysis
# =========================================================
with tab3:
    st.subheader("📖 Theoretical Foundations & Mathematical Formulations")

    st.markdown("### 1. Integer Linear Programming (ILP) Formulation")
    st.markdown("The **Weighted Set Cover** facility problem is mathematically formulated as an integer program:")

    st.latex(r"""
    \begin{aligned}
    \min_{x} \quad & \sum_{i=1}^{m} c_i x_i \\
    \text{subject to} \quad & \sum_{i: z_j \in S_i} x_i \ge 1, \quad \forall j \in \{1, \dots, n\} \\
    & x_i \in \{0, 1\}, \quad \forall i \in \{1, \dots, m\}
    \end{aligned}
    """)

    st.markdown(r"""
    **Where:**
    * $x_i = 1$ indicates that candidate facility $F_i$ is opened ($x_i = 0$ otherwise).
    * $c_i > 0$ is the positive installation cost of facility site $F_i$.
    * $S_i = \{z_j \in U \mid \|F_i - z_j\|_2 \le R_i\}$ is the spatial coverage disk of facility $F_i$.
    * $\sum_{i: z_j \in S_i} x_i \ge 1$ enforces that every demand zone $z_j$ is serviced by at least one installed facility.
    """)

    st.markdown("---")

    theo_col1, theo_col2 = st.columns(2)

    with theo_col1:
        st.markdown("### 2. Chvátal's Greedy Approximation Rule")
        st.markdown(
            r"**Greedy Selection Invariant:** At each iteration $t$, let $U_{\text{rem}}$ be the set of "
            r"currently uncovered demand zones. Select candidate facility $i^*$ that minimizes the marginal "
            r"cost-effectiveness ratio:"
        )

        st.latex(r"\alpha_i = \frac{c_i}{|S_i \cap U_{\text{rem}}|}")

        st.markdown("""
        ```python
        # Pure Python Implementation of Chvátal's Rule
        U_rem = set(range(n))
        selected = []
        while U_rem:
            i_star = min(
                (i for i in candidates if i not in selected),
                key=lambda i: c[i] / len(S[i] & U_rem) if (S[i] & U_rem) else float('inf')
            )
            selected.append(i_star)
            U_rem -= S[i_star]
        ```
        """)

        st.markdown(r"""
        **Theoretical Approximation Guarantee (Chvátal, 1979):**  
        The greedy algorithm achieves an approximation factor bounded by the $n$-th harmonic number $H_n$:
        """)

        st.latex(r"\rho = \frac{\sum_{i \in I_{\text{Greedy}}} c_i}{C_{\text{OPT}}} \le H_n = \sum_{k=1}^{n} \frac{1}{k} \le \ln n + 1")

    with theo_col2:
        st.markdown("### 3. Dual-Fitting Proof Intuition")
        st.markdown("Consider the linear programming (LP) relaxation dual:")

        st.latex(r"""
        \begin{aligned}
        \max_{y} \quad & \sum_{j=1}^{n} y_j \\
        \text{subject to} \quad & \sum_{j \in S_i} y_j \le c_i, \quad \forall i \in \{1, \dots, m\} \\
        & y_j \ge 0, \quad \forall j \in \{1, \dots, n\}
        \end{aligned}
        """)

        st.markdown(r"""
        When Greedy picks facility $i^*$ covering new elements $\Delta U = S_{i^*} \cap U_{\text{rem}}$, 
        we charge each newly covered element $z \in \Delta U$ a price:
        """)

        st.latex(r"p(z) = \frac{c_{i^*}}{|\Delta U|}")

        st.markdown(r"""
        By design, the total greedy cost equals the sum of element prices:
        """)

        st.latex(r"\sum_{i \in I_{\text{Greedy}}} c_i = \sum_{j=1}^{n} p(z_j)")

        st.markdown(r"""
        For any candidate set $S_k$, ordering its elements in reverse order of coverage reveals:
        """)

        st.latex(r"\sum_{z \in S_k} p(z) \le c_k \cdot \sum_{r=1}^{|S_k|} \frac{1}{r} = c_k H_{|S_k|} \le c_k H_n")

        st.markdown(r"""
        Setting dual variables $y_j = \frac{p(z_j)}{H_n}$ yields a feasible dual solution, proving by LP weak duality:
        """)

        st.latex(r"\text{Cost}(\text{Greedy}) = \sum p(z_j) = H_n \sum y_j \le H_n \cdot C_{\text{OPT}}")

    st.markdown("---")

    st.subheader("4. Asymptotic Complexity Comparison")

    complexity_df = pd.DataFrame([
        {
            "Algorithm": "Exhaustive Subset Enumeration",
            "Paradigm": "Brute-Force Combinatorial",
            "Worst-Case Time": "O(2^m · n)",
            "Space Complexity": "O(m + n)",
            "Optimality Guarantee": "Exact Global Optimum (1.00)",
            "Scalability Limit": "m <= 18 facilities"
        },
        {
            "Algorithm": "Branch-and-Bound (BnB)",
            "Paradigm": "Exact Pruned State-Space Tree",
            "Worst-Case Time": "O(2^m · n)",
            "Space Complexity": "O(m + n)",
            "Optimality Guarantee": "Exact Global Optimum (1.00)",
            "Scalability Limit": "m <= 24 facilities"
        },
        {
            "Algorithm": "Chvátal's Greedy Set Cover",
            "Paradigm": "Marginal Ratio Greedy",
            "Worst-Case Time": "O(m · n)",
            "Space Complexity": "O(m · n)",
            "Optimality Guarantee": "rho <= H_n ≈ ln(n) + 0.577",
            "Scalability Limit": "m, n >= 100,000"
        },
        {
            "Algorithm": "Greedy + 1-Swap Local Search",
            "Paradigm": "Approximation + Post-Optimization",
            "Worst-Case Time": "O(m · n + m^2 · n)",
            "Space Complexity": "O(m · n)",
            "Optimality Guarantee": "Empirical rho -> 1.00",
            "Scalability Limit": "m, n >= 5,000"
        }
    ])

    st.table(complexity_df)

    st.markdown("""
    > [!NOTE] **DAA Project Takeaway:**
    > While Weighted Set Cover is NP-hard, Chvátal's greedy heuristic solves massive real-world municipal 
    > facility planning instances in sub-millisecond execution times while empirically maintaining 
    > near-optimal installation costs ($\rho \approx 1.00 - 1.15$ in practice), far outperforming the 
    > theoretical worst-case harmonic ceiling $H_n$.
    """)

