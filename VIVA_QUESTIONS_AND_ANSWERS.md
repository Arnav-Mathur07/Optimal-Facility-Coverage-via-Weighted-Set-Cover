# DAA PBL Viva Defense Cheat Sheet & Flashcard Guide
## Weighted Set Cover for Optimal Facility Coverage

This document is your **quick-reference master guide** for viva examination. Memorize these key concepts, definitions, equations, and answers.

---

## ⚡ 60-Second Elevator Pitch
> *"My project investigates the NP-hard Weighted Set Cover problem applied to municipal facility location. Given $n$ demand zones and $m$ candidate facility sites with opening costs and coverage radii, we seek the minimum-cost facility subset that guarantees 100% coverage. We implemented four algorithms from scratch in pure Python/NumPy: an Exhaustive Combinatorial Baseline $O(2^m \cdot n)$, a Branch-and-Bound exact solver with cost and suffix reachability pruning, Chvátal's Greedy Approximation Algorithm $O(m \cdot n)$ with proven approximation ratio $\rho \le H_n$, and a 1-Opt Local Search post-processing refinement. We verified mathematical correctness through 6 automated PyTest invariant tests, built a multi-tab Streamlit analytical dashboard, and created a cinematic interactive canvas simulation."*

---

## 📌 Core Formula Cheat Sheet

| Concept | Mathematical Equation | Meaning |
| :--- | :--- | :--- |
| **Primal Objective** | $\min \sum_{i=1}^m c_i x_i$ subject to $\sum_{i: z_j \in S_i} x_i \ge 1, x_i \in \{0, 1\}$ | Minimize total facility opening cost such that every zone $z_j$ is covered. |
| **Dual LP Objective** | $\max \sum_{j=1}^n y_j$ subject to $\sum_{j \in S_i} y_j \le c_i, y_j \ge 0$ | Maximize virtual price charged to zones without exceeding any facility budget $c_i$. |
| **Greedy Selection Rule** | $i^* = \arg\min_{i \notin I} \alpha_i = \arg\min_{i \notin I} \frac{c_i}{\|S_i \cap U_{\text{rem}}\|}$ | Pick the facility giving the lowest marginal cost per newly covered demand zone. |
| **Harmonic Bound** | $H_n = \sum_{k=1}^n \frac{1}{k} = 1 + \frac{1}{2} + \frac{1}{3} + \dots + \frac{1}{n} \approx \ln n + \gamma$ | Theoretical worst-case ceiling on the approximation ratio ($\gamma \approx 0.5772$). |
| **Approximation Ratio** | $\rho = \frac{\text{Cost}(\text{Greedy})}{C_{\text{OPT}}} \ge 1.0$ | How many times more expensive the heuristic solution is compared to the exact global optimum. |
| **Bitwise Target Mask** | $\text{target} = (1 \ll n) - 1$ | Binary integer with first $n$ bits set to 1, representing 100% universe coverage. |

---

## 🎯 Top 15 Viva Questions & High-Scoring Model Answers

### Category 1: Problem Complexity & Foundations

#### Q1: Why is Weighted Set Cover classified as NP-hard?
* **Answer:** "Richard Karp in 1972 proved that the Vertex Cover problem—which is NP-complete—can be reduced in polynomial time to the Set Cover problem. Since Vertex Cover $\le_P$ Set Cover, Set Cover is at least as hard as Vertex Cover. In our facility coverage problem, deciding whether there exists a facility subset of cost at most $K$ covering all zones is NP-complete, making the minimization version NP-hard."

#### Q2: Can we solve Weighted Set Cover in polynomial time if $P \neq NP$?
* **Answer:** "No. If $P \neq NP$, no polynomial-time exact algorithm exists for any NP-hard problem because the search space of combinations grows exponentially as $O(2^m)$."

#### Q3: What is the difference between a heuristic and an approximation algorithm?
* **Answer:** "A heuristic is a rule of thumb (like greedy or genetic algorithms) that usually finds good solutions quickly, but offers **no mathematical guarantee** on solution quality. An approximation algorithm is a polynomial-time algorithm backed by a **rigorous mathematical proof** guaranteeing that its solution cost will never exceed $\rho \times C_{\text{OPT}}$ on any instance."

#### Q4: Why can't we achieve a constant-factor approximation (like 2-approximation) for Set Cover?
* **Answer:** "Because of **Feige's Inapproximability Theorem (1998)**. Feige proved that unless NP problems can be solved in $DTIME(n^{\log \log n})$, no polynomial-time algorithm can approximate Set Cover within $(1 - \epsilon)\ln n$ for any $\epsilon > 0$. Hence, Chvátal's $H_n \approx \ln n$ bound is asymptotically optimal—it hits the theoretical threshold of computational hardness."

---

### Category 2: Algorithm Mechanics & Proofs

#### Q5: Explain the Dual-Fitting proof of the harmonic bound $H_n$ in simple steps.
* **Answer:** 
  1. "When Greedy selects facility $i^*$ covering $\Delta U$ new zones, we charge each newly covered zone a price $p(z) = c_{i^*} / |\Delta U|$. The total greedy cost equals $\sum p(z)$."
  2. "For any candidate set $S_k$, sorting its elements in reverse order of coverage proves $\sum_{z \in S_k} p(z) \le c_k H_{|S_k|} \le c_k H_n$."
  3. "If we set dual variables $y_j = p(z_j) / H_n$, the budget constraints $\sum_{j \in S_k} y_j \le c_k$ are satisfied, making $y$ a valid feasible solution to the Dual Linear Program."
  4. "By weak LP duality, the dual objective is $\le C_{\text{OPT}}$. Multiplying by $H_n$ gives $\text{Cost}(\text{Greedy}) \le H_n \cdot C_{\text{OPT}}$."

#### Q6: How does the Branch-and-Bound algorithm prune its search tree?
* **Answer:** "It constructs a binary inclusion/exclusion tree and uses two admissible pruning checks:
  1. **Cost-Bound Pruning:** Since all facility costs $c_i > 0$, if $\text{current\_cost} \ge \text{best\_cost}$, we immediately terminate the branch because adding more facilities can never lower the cost.
  2. **Suffix Reachability Pruning:** We precompute bitwise suffix unions $\text{suffix\_union}[d] = \bigcup_{i=d}^{m-1} S_i$. If $(\text{current\_mask} \mid \text{suffix\_union}[d]) \neq \text{target\_mask}$, even selecting all remaining facilities cannot achieve 100% coverage. We prune the branch immediately."

#### Q7: Why did you use integer bitmasks instead of Python sets or boolean arrays?
* **Answer:** "Standard Python set unions (`set.union()`) allocate heap memory and iterate through hash tables. In contrast, integer bitmasks map each zone to a single binary bit. Checking 100% coverage is a single CPU equality check (`mask == target_mask`), and combining facilities is a single-cycle bitwise OR (`mask | fac_mask`). This allows us to evaluate over 300,000 combinations per second in pure Python without C extensions."

---

### Category 3: Heuristics, Post-Processing & Adversarial Cases

#### Q8: What is 1-Opt Local Search, and why is it needed after Greedy?
* **Answer:** "Greedy algorithms make irrevocable decisions. Early choices made when many zones were uncovered might become completely redundant after later facilities are added. Our 1-Opt refinement runs:
  1. *Redundancy Pruning:* Sorts selected facilities in descending order of cost and safely removes any facility whose removal leaves coverage at 100%.
  2. *1-Swap Search:* Replaces a selected facility with a cheaper unselected candidate if feasibility is preserved.
  This tightens the empirical ratio from $\approx 1.12$ down to $\approx 1.02$ and raises the optimal hit rate to 70%."

#### Q9: How does your Adversarial Harmonic generator trick Chvátal's Greedy algorithm?
* **Answer:** "It constructs the classic Lovász-Chvátal worst-case instance. One central optimal hub covers all $n$ zones at cost $1.05$. Then, $m-1$ deceptive subsets cover geometrically decreasing groups of zones ($n/2, n/4, \dots$). The costs of the deceptive subsets are calibrated so their marginal cost ratio is slightly cheaper than the hub at each step. Greedy picks all $m-1$ deceptive subsets one by one paying $\approx H_n$, while the exact solver picks only the central hub. This yields $\rho \approx 2.82 > 1.30$, experimentally proving that the $H_n$ ceiling is reachable."

#### Q10: Why is there such a large gap between the theoretical bound ($H_{30} \approx 3.99$) and empirical ratios on random instances ($\rho \approx 1.05 - 1.12$)?
* **Answer:** "$H_n$ is a worst-case mathematical bound designed to hold under adversarial sabotage. On realistic geographic grids, facility coverage disks overlap smoothly. Greedy decisions reflect genuine spatial density rather than deceptive traps, so greedy performs near-optimally in practice."

---

### Category 4: Engineering, Architecture & Verification

#### Q11: How do you guarantee 100% feasibility during random instance generation?
* **Answer:** "In `_ensure_full_coverage()` in `generator.py`, we compute the coverage matrix. If any demand zone is uncovered, the engine finds the nearest facility, expands its radius ($R_{i^*} = \text{dist} + 1.5$), and increases its cost proportional to area. This guarantees that $\bigcup_{i=1}^m S_i = U$ strictly holds on 100% of generated instances."

#### Q12: What automated tests did you write, and what invariants do they verify?
* **Answer:** "We implemented 6 automated test suites in `tests/test_solvers.py`:
  1. 100% feasibility on uniform instances.
  2. 100% feasibility on clustered urban instances.
  3. Exact cost equivalence between Branch-and-Bound and Brute Force with state reduction.
  4. Strict optimality invariant: $\text{cost}(\text{Greedy}) \ge \text{cost}(\text{OPT})$ across 50 randomized instances with zero violations ($\rho \ge 1.00$).
  5. Adversarial harmonic bound test: $\rho > 1.30$ and $\rho \le H_n$.
  6. Monotonic coverage progression in the greedy trace."

#### Q13: Compare the time and space complexity of all implemented algorithms.
* **Answer:** 
  * Exhaustive Brute-Force: Time $O(2^m \cdot n)$, Space $O(m + n)$.
  * Branch-and-Bound: Time $O(2^m \cdot n)$ worst-case, $O(1.3^m \cdot n)$ average, Space $O(m + n)$.
  * Chvátal's Greedy: Time $O(m \cdot n)$, Space $O(m \cdot n)$.
  * Refined Greedy (1-Opt): Time $O(m \cdot n + m^2 \cdot n)$, Space $O(m \cdot n)$.

#### Q14: What real-world applications does this project model?
* **Answer:** "Emergency medical services (positioning ambulances/hospitals to cover all districts within response-time radii), telecommunications (5G cell tower placement minimizing infrastructure cost), municipal services (fire stations and waste management depots), and EV fast-charging network planning along highway corridors."

#### Q15: What technologies power the project's interfaces?
* **Answer:** 
  * Backend: Python 3.11 with NumPy and Pandas.
  * Analytical Dashboard: Streamlit with Plotly interactive 2D spatial maps and SVG charts (Port 8501).
  * Web Application: Standalone single-file HTML5 Canvas lab with client-side JavaScript algorithm engines, KaTeX mathematical typesetting, and MotionSites cinematic video art direction (Port 8080).

---

## 🏆 Viva Pro-Tips ("Examiner Impressions")
1. **Always emphasize "Pure Python / No Black-Box Solvers":** Mention that we implemented the branch-and-bound and greedy algorithms from scratch using bitwise arithmetic so step counts and runtimes reflect real algorithmic complexity.
2. **Distinguish Primal vs Dual:** If asked about $H_n$, immediately mention **"Dual Fitting and Weak LP Duality"**—professors love this theoretical rigor.
3. **Know the Asymptotics:** Memorize that Brute Force is $O(2^m \cdot n)$ while Greedy is $O(m \cdot n)$.
4. **Mention the Inapproximability Theorem:** Citing **Feige's 1998 proof** shows deep mastery beyond standard textbook material.
