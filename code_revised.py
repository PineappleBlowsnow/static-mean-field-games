"""
Two-way comparison for the SMFG full-feedback setting on 3 Problems (Linear, KL, BB).

Algorithms compared
-------------------
1. TRPA-Full (paper-style full-feedback Algorithm 1 with fixed tau = N^{-1/4})
2. A-TRPA-Full (same structure, but with annealed regularization tau_t)

Main changes vs the original script
-----------------------------------
1. KL payoff can be sign-flipped through CONFIG["kl_payoff_sign"].
   The default below is -1.0, matching the decreasing-payoff convention.
   Set it to +1.0 if you want the exact Appendix-F KL formula as printed in the paper.
2. Scaling plot includes both N^{-1/4} and N^{-1/2} reference lines.
3. Exploitability is evaluated by the finite-N definition from the paper, using a Monte Carlo
   estimator of V^i(pi^i, pi^{-i}) and max_{pi'} V^i(pi', pi^{-i}).
4. Learning curves include a shaded 95% confidence interval over run seeds.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1) Configuration
# ============================================================

OUTPUT_DIR = Path(".")

CONFIG = {
    "K": 5,
    "N_values": [20,100,500,1000],
    "T": 10000,
    "noise_std": 0.10,
    "problem_seed": 7,
    "run_seeds": [0, 1, 2, 3, 4],
    "theory_consistent_sign": True,
    "eta0_annealed": 0.5,

    # KL sign convention:
    #   +1.0: exact Appendix-F KL formula as printed in the paper.
    #   -1.0: sign-flipped payoff, so larger occupancy lowers the payoff.
    "kl_payoff_sign": -1.0,

    # Exploitability evaluation:
    #   "finite_mc" follows the paper's finite-N exploitability definition, estimated by MC.
    #   "mean_field_proxy" recovers the original cheap approximation F(mean_policy).
    "exploitability_mode": "finite_mc",
    "exploitability_mc_samples": 128,

    # The finite-N MC estimator is substantially more expensive than the MF proxy.
    # Evaluate every eval_every rounds and carry the latest estimate forward in between.
    # Set to 1 if you want every round evaluated exactly by the MC estimator.
    "exploitability_eval_every": 10,

    # Plotting
    "ci_z": 1.96,
    "smoothing_window": 25,
}

# ============================================================
# 2) Helper functions & Payoff Definitions
# ============================================================

def project_rows_to_simplex(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    sorted_rows = np.sort(matrix, axis=1)[:, ::-1]
    cumulative = np.cumsum(sorted_rows, axis=1) - 1.0
    indices = np.arange(1, matrix.shape[1] + 1, dtype=float)

    condition = sorted_rows - cumulative / indices > 0
    rho = condition.sum(axis=1) - 1
    theta = cumulative[np.arange(matrix.shape[0]), rho] / (rho + 1.0)

    return np.maximum(matrix - theta[:, None], 0.0)


def sample_actions_from_policies(policies: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cdf = np.cumsum(policies, axis=1)
    # Numerical guard: make the last CDF entry exactly one.
    cdf[:, -1] = 1.0
    u = rng.random((policies.shape[0], 1))
    return (u > cdf).sum(axis=1)


def get_problem(
    name: str,
    K: int,
    seed: int,
    theory_consistent_sign: bool = True,
    kl_payoff_sign: float = -1.0,
):
    """
    Returns a payoff operator function F_func(mu) for the requested problem.

    Notes on signs:
    - Linear uses A = -S + X when theory_consistent_sign=True, so the symmetric part
      is decreasing/congestive.
    - KL uses kl_payoff_sign. The paper's appendix writes +grad KL; the sign-flipped
      version is useful if you want congestion-style decreasing payoff.
    """
    rng = np.random.default_rng(seed)

    if name == "Linear":
        M = rng.normal(size=(K, K))
        S = (M.T @ M) / K + 0.25 * np.eye(K)
        U = rng.uniform(0.0, 1.0, size=(K, K))
        X = (U - U.T) / 2.0
        b = rng.uniform(0.0, 1.0, size=K)
        A = (-S + X) if theory_consistent_sign else (S + X)
        return lambda mu: A @ mu + b

    if name == "KL":
        mu_ref = rng.uniform(0.0, 1.0, size=K)
        mu_ref /= mu_ref.sum()
        gamma = 0.1

        def F_kl(mu: np.ndarray) -> np.ndarray:
            mu = np.asarray(mu, dtype=float)
            raw = gamma * np.log((gamma * mu + (1.0 - gamma) * mu_ref) / mu_ref) + gamma
            return kl_payoff_sign * raw

        return F_kl

    if name == "BB":
        x_bar = K // 2
        a_vals = np.arange(K)
        alpha = 1.0
        base_payoff = 1.0 - np.abs(a_vals - x_bar) / K
        return lambda mu: base_payoff - alpha * np.log(1.0 + mu)

    raise ValueError(f"Unknown problem name: {name!r}. Expected one of: Linear, KL, BB")


def compute_exploitability_mean_field_proxy(policies: np.ndarray, F_func) -> np.ndarray:
    """
    Cheap proxy used by the original script:
        max_a F(mean_policy, a) - <pi_i, F(mean_policy)>.

    This is NOT the finite-N exploitability definition in the paper; it is only a
    mean-field approximation.
    """
    mean_policy = policies.mean(axis=0)
    F_val = F_func(mean_policy)
    current_values = np.sum(policies * F_val, axis=1)
    best_values = np.max(F_val)
    return np.maximum(best_values - current_values, 0.0)


def compute_exploitability_finite_mc(
    policies: np.ndarray,
    F_func,
    rng: np.random.Generator,
    n_samples: int = 128,
) -> np.ndarray:
    """
    Monte Carlo estimator of the finite-N exploitability from Definition 1:

        E_exp^i({pi^j}) = max_{pi' in Delta_A} V^i(pi', pi^{-i})
                          - V^i(pi^i, pi^{-i}).

    Because V^i is linear in the deviating policy pi', the best response over Delta_A
    is attained by a pure action. For each MC sample, we sample all agents' actions,
    remove agent i's sampled action to obtain an estimate of the other agents' occupancy,
    then test all pure deviations a in {1,...,K}.

    Complexity is roughly O(n_samples * N * K) plus O(n_samples * K^2) payoff calls.
    """
    policies = np.asarray(policies, dtype=float)
    N, K = policies.shape

    current_value_acc = np.zeros(N, dtype=float)
    br_value_acc = np.zeros((N, K), dtype=float)

    for _ in range(n_samples):
        actions = sample_actions_from_policies(policies, rng)
        counts = np.bincount(actions, minlength=K).astype(float)

        # Current-play value contribution: F(mu_hat)(a_i).
        F_current = F_func(counts / N)
        current_value_acc += F_current[actions]

        # Best response contribution. For a fixed sampled old action b, all agents
        # with action b share the same counterfactual payoff table for this MC sample.
        br_by_old_action = np.zeros((K, K), dtype=float)
        for old_action in range(K):
            if counts[old_action] <= 0:
                continue
            counts_without_i = counts.copy()
            counts_without_i[old_action] -= 1.0
            for candidate_action in range(K):
                candidate_counts = counts_without_i.copy()
                candidate_counts[candidate_action] += 1.0
                br_by_old_action[old_action, candidate_action] = (
                    F_func(candidate_counts / N)[candidate_action]
                )

        br_value_acc += br_by_old_action[actions]

    current_values = current_value_acc / n_samples
    br_values = br_value_acc / n_samples
    best_values = br_values.max(axis=1)

    # True exploitability is nonnegative; clipping removes small MC noise artifacts.
    return np.maximum(best_values - current_values, 0.0)


def compute_exploitability(
    policies: np.ndarray,
    F_func,
    rng: np.random.Generator,
    mode: str = "finite_mc",
    n_mc_samples: int = 128,
) -> np.ndarray:
    if mode == "finite_mc":
        return compute_exploitability_finite_mc(policies, F_func, rng, n_samples=n_mc_samples)
    if mode == "mean_field_proxy":
        return compute_exploitability_mean_field_proxy(policies, F_func)
    raise ValueError(f"Unknown exploitability mode: {mode!r}")


def one_round_full_feedback(policies: np.ndarray, F_func, noise_std: float, rng: np.random.Generator):
    N, K = policies.shape
    actions = sample_actions_from_policies(policies, rng)
    mu_hat = np.bincount(actions, minlength=K) / N
    mean_payoff = F_func(mu_hat)
    rewards = mean_payoff + rng.normal(scale=noise_std, size=(N, K))
    return rewards, mu_hat


def summarize_runs(curve_matrix: np.ndarray, ci_z: float = 1.96) -> dict:
    curve_matrix = np.asarray(curve_matrix, dtype=float)
    n_runs = curve_matrix.shape[0]
    last100 = curve_matrix[:, -100:].mean(axis=1)
    std_last100 = float(last100.std(ddof=1)) if n_runs > 1 else 0.0
    sem_last100 = std_last100 / np.sqrt(max(n_runs, 1))
    return {
        "final_avg_last100_max_exploitability": float(last100.mean()),
        "std_across_runs_last100": std_last100,
        "ci95_across_runs_last100": float(ci_z * sem_last100),
        "auc_mean_max_exploitability": float(curve_matrix.mean()),
        "best_seen_mean_max_exploitability": float(curve_matrix.min(axis=1).mean()),
        "n_runs": int(n_runs),
    }


def maybe_smooth(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values
    return pd.Series(values).rolling(window=window, min_periods=1, center=False).mean().to_numpy()

# ============================================================
# 3) Algorithms
# ============================================================

def _append_history_record(history, t, exploitabilities, evaluated_now: bool):
    history.append({
        "t": t + 1,
        "max_exploitability": float(exploitabilities.max()),
        "mean_exploitability": float(exploitabilities.mean()),
        "exploitability_evaluated_now": bool(evaluated_now),
    })


def run_trpa_full_original(
    F_func,
    K: int,
    N: int,
    T: int,
    noise_std: float,
    seed: int,
    exploitability_mode: str = "finite_mc",
    exploitability_mc_samples: int = 128,
    exploitability_eval_every: int = 10,
):
    """
    Paper-style full-feedback TRPA:
        pi_{t+1} = Proj((1 - tau * eta_t) pi_t + eta_t r_t)
    with
        tau = N^{-1/4},
        eta_t = 1 / (tau * (t + 2)).
    """
    rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(seed + 10_000_000)

    tau = N ** (-0.25)
    policies = np.ones((N, K)) / K

    history = []
    last_exploitabilities = None
    eval_every = max(int(exploitability_eval_every), 1)

    for t in range(T):
        rewards, _ = one_round_full_feedback(policies, F_func, noise_std, rng)

        eta_t = 1.0 / (tau * (t + 2.0))
        policies = project_rows_to_simplex((1.0 - tau * eta_t) * policies + eta_t * rewards)

        evaluated_now = (t % eval_every == 0) or (t == T - 1) or (last_exploitabilities is None)
        if evaluated_now:
            last_exploitabilities = compute_exploitability(
                policies,
                F_func,
                eval_rng,
                mode=exploitability_mode,
                n_mc_samples=exploitability_mc_samples,
            )

        _append_history_record(history, t, last_exploitabilities, evaluated_now)

    return policies, pd.DataFrame(history)


def run_trpa_full_annealed(
    F_func,
    K: int,
    N: int,
    T: int,
    noise_std: float,
    seed: int,
    eta0: float = 0.5,
    exploitability_mode: str = "finite_mc",
    exploitability_mc_samples: int = 128,
    exploitability_eval_every: int = 10,
):
    """
    Same per-round TRPA structure, but with annealed regularization:
        tau_t = tau0 / sqrt(t+1),
        eta_t = eta0 / sqrt(t+1).

    This is not the paper's theorem setting; it is an extra heuristic comparison.
    """
    rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(seed + 20_000_000)

    tau0 = N ** (-0.25)
    policies = np.ones((N, K)) / K

    history = []
    last_exploitabilities = None
    eval_every = max(int(exploitability_eval_every), 1)

    for t in range(T):
        rewards, _ = one_round_full_feedback(policies, F_func, noise_std, rng)

        #tau_t = 1 / pow(t+1.0,1/7) # tau0 / pow(t+1.0,1/7)
        #eta_t = 1 / pow(t+1.0,5/7) # eta0 / pow(t+1.0,5/7)
        
        tau_t = tau0 / np.sqrt(t + 1.0)
        eta_t = eta0 / np.sqrt(t + 1.0)
        policies = project_rows_to_simplex((1.0 - tau_t * eta_t) * policies + eta_t * rewards)

        evaluated_now = (t % eval_every == 0) or (t == T - 1) or (last_exploitabilities is None)
        if evaluated_now:
            last_exploitabilities = compute_exploitability(
                policies,
                F_func,
                eval_rng,
                mode=exploitability_mode,
                n_mc_samples=exploitability_mc_samples,
            )

        _append_history_record(history, t, last_exploitabilities, evaluated_now)

    return policies, pd.DataFrame(history)

# ============================================================
# 4) Benchmark
# ============================================================

def benchmark_all(config: dict):
    summary_rows = []
    curve_rows = []
    problems = ["Linear", "KL", "BB"]

    for prob_name in problems:
        print(f"\n====================== Problem: {prob_name} ======================")
        F_func = get_problem(
            prob_name,
            config["K"],
            config["problem_seed"],
            config["theory_consistent_sign"],
            kl_payoff_sign=config.get("kl_payoff_sign", -1.0),
        )

        algo_specs = [
            ("TRPA-Full", lambda N, seed: run_trpa_full_original(
                F_func=F_func,
                K=config["K"],
                N=N,
                T=config["T"],
                noise_std=config["noise_std"],
                seed=seed,
                exploitability_mode=config.get("exploitability_mode", "finite_mc"),
                exploitability_mc_samples=config.get("exploitability_mc_samples", 128),
                exploitability_eval_every=config.get("exploitability_eval_every", 10),
            )),
            ("A-TRPA-Full", lambda N, seed: run_trpa_full_annealed(
                F_func=F_func,
                K=config["K"],
                N=N,
                T=config["T"],
                noise_std=config["noise_std"],
                seed=seed,
                eta0=config["eta0_annealed"],
                exploitability_mode=config.get("exploitability_mode", "finite_mc"),
                exploitability_mc_samples=config.get("exploitability_mc_samples", 128),
                exploitability_eval_every=config.get("exploitability_eval_every", 10),
            )),
        ]

        for N in config["N_values"]:
            print(f"--- Running N = {N} ---")
            curves_by_algo = {name: [] for name, _ in algo_specs}

            for seed in config["run_seeds"]:
                line = [f"  seed={seed}"]
                for name, runner in algo_specs:
                    _, history = runner(N, seed)
                    curves_by_algo[name].append(history["max_exploitability"].to_numpy())
                    line.append(f"{name} final={history['max_exploitability'].iloc[-1]:.4f}")
                print(" | ".join(line))

            # Aggregate stats across seeds.
            for name, _ in algo_specs:
                curves = np.vstack(curves_by_algo[name])
                summary = summarize_runs(curves, ci_z=config.get("ci_z", 1.96))
                summary_rows.append({"problem": prob_name, "N": N, "algorithm": name, **summary})

                mean_curve = curves.mean(axis=0)
                std_curve = curves.std(axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros_like(mean_curve)
                ci_curve = config.get("ci_z", 1.96) * std_curve / np.sqrt(curves.shape[0])
                lower_curve = np.maximum(mean_curve - ci_curve, 0.0)
                upper_curve = mean_curve + ci_curve

                for t in range(config["T"]):
                    curve_rows.append({
                        "problem": prob_name,
                        "N": N,
                        "t": t + 1,
                        "algorithm": name,
                        "mean_max_exploitability": float(mean_curve[t]),
                        "std_max_exploitability": float(std_curve[t]),
                        "ci95_lower_max_exploitability": float(lower_curve[t]),
                        "ci95_upper_max_exploitability": float(upper_curve[t]),
                    })

    return pd.DataFrame(summary_rows), pd.DataFrame(curve_rows)

# ============================================================
# 5) Plotting
# ============================================================

def plot_learning_curves(curves_df: pd.DataFrame, save_path: Path, smoothing_window: int = 25):
    """
    Paper-style learning curves:
    - log-log axes,
    - smoothed max exploitability,
    - shaded 95% CI over seeds.
    """
    problems = ["Linear", "KL", "BB"]
    algos = ["TRPA-Full", "A-TRPA-Full"]

    fig, axes = plt.subplots(len(algos), len(problems), figsize=(18, 8), squeeze=False)
    eps = 1e-12

    for i, algo in enumerate(algos):
        for j, prob in enumerate(problems):
            ax = axes[i, j]
            subset = curves_df[(curves_df["problem"] == prob) & (curves_df["algorithm"] == algo)]

            for N in sorted(subset["N"].unique(), reverse=True):
                line = subset[subset["N"] == N].sort_values("t")
                x = line["t"].to_numpy()
                y = maybe_smooth(line["mean_max_exploitability"].to_numpy(), smoothing_window)
                lo = maybe_smooth(line["ci95_lower_max_exploitability"].to_numpy(), smoothing_window)
                hi = maybe_smooth(line["ci95_upper_max_exploitability"].to_numpy(), smoothing_window)

                y = np.maximum(y, eps)
                lo = np.maximum(lo, eps)
                hi = np.maximum(hi, eps)

                plotted_line, = ax.plot(x, y, label=f"N={N}", linewidth=1.5)
                ax.fill_between(x, lo, hi, alpha=0.18, color=plotted_line.get_color(), linewidth=0)

            ax.set_title(f"{prob} | {algo}")
            ax.set_xlabel("Time (log scale)")
            ax.set_ylabel("Maximum exploitability")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.25)

            if i == 0 and j == 0:
                ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_scaling_law_vs_N(summary_df: pd.DataFrame, save_path: Path):
    """
    Plots Final Exploitability vs N with both reference slopes:
    - O(N^{-1/4}): finite-learning bias scale for monotone TRPA-Full after choosing tau=N^{-1/4}.
    - O(N^{-1/2}): MF-NE finite-agent approximation scale from Theorem 1.
    """
    problems = ["Linear", "KL", "BB"]
    algos = ["TRPA-Full", "A-TRPA-Full"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, prob in enumerate(problems):
        ax = axes[i]
        subset = summary_df[summary_df["problem"] == prob]

        # 1. Plot empirical final performance with CI error bars.
        for algo in algos:
            algo_data = subset[subset["algorithm"] == algo].sort_values("N")
            x = algo_data["N"].to_numpy()
            y = algo_data["final_avg_last100_max_exploitability"].to_numpy()
            yerr = algo_data["ci95_across_runs_last100"].to_numpy()
            ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2, capsize=3, label=algo)

        # 2. Plot reference slopes anchored to TRPA-Full at the smallest N.
        anchor_algo_data = subset[subset["algorithm"] == "TRPA-Full"].sort_values("N")
        if len(anchor_algo_data) > 0:
            anchor = float(anchor_algo_data["N"].iloc[0])
            anchor_val = float(anchor_algo_data["final_avg_last100_max_exploitability"].iloc[0])
            n_range = np.array(sorted(summary_df["N"].unique()), dtype=float)

            c_quarter = anchor_val * (anchor ** 0.25)
            ref_quarter = c_quarter / (n_range ** 0.25)
            ax.plot(n_range, ref_quarter, linestyle="--", linewidth=2, label=r"Reference $N^{-1/4}$")

            c_half = anchor_val * np.sqrt(anchor)
            ref_half = c_half / np.sqrt(n_range)
            ax.plot(n_range, ref_half, linestyle=":", linewidth=2.5, label=r"Reference $N^{-1/2}$")

        ax.set_title(f"{prob} - Scaling with N")
        ax.set_xlabel("Number of Agents ($N$)")
        ax.set_ylabel("Final Exploitability")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(CONFIG["N_values"])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_final_bars(summary_df: pd.DataFrame, save_path: Path):
    """
    Plots a bar chart comparing final performance for each algorithm,
    grouped by Problem and N.
    """
    plot_df = summary_df.copy()
    plot_df["Prob_N"] = plot_df["problem"] + "_N" + plot_df["N"].astype(str)

    pivot = plot_df.pivot(
        index="Prob_N",
        columns="algorithm",
        values="final_avg_last100_max_exploitability",
    )

    pivot = pivot.reindex(sorted(pivot.index, key=lambda x: (x.split("_")[0], int(x.split("_N")[1]))))

    ax = pivot.plot(kind="bar", figsize=(12, 5), rot=45)
    ax.set_title("Final Performance Comparison")
    ax.set_xlabel("Problem & Agent Count")
    ax.set_ylabel("Avg Max Exploitability (Last 100 rounds)")
    ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()

# ============================================================
# 6) Main
# ============================================================

def main():
    print("Starting 3-Problem SMFG Benchmark...")
    print("Algorithms: TRPA-Full vs A-TRPA-Full")
    print(f"Exploitability mode: {CONFIG['exploitability_mode']}")
    print(f"KL payoff sign: {CONFIG['kl_payoff_sign']}")
    print("-" * 50)

    summary_df, curves_df = benchmark_all(CONFIG)

    # Define paths
    summary_path = OUTPUT_DIR / "smfg_summary.csv"
    curves_path = OUTPUT_DIR / "smfg_curves.csv"
    script_config_path = OUTPUT_DIR / "smfg_config.json"

    curves_plot_path = OUTPUT_DIR / "smfg_learning_curves.png"
    scaling_plot_path = OUTPUT_DIR / "smfg_scaling_vs_N.png"
    bars_plot_path = OUTPUT_DIR / "smfg_final_bars.png"

    # Save data
    summary_df.to_csv(summary_path, index=False)
    curves_df.to_csv(curves_path, index=False)
    script_config_path.write_text(json.dumps(CONFIG, indent=2))

    # Generate all plots
    print("\nGenerating plots...")
    plot_learning_curves(curves_df, curves_plot_path, smoothing_window=CONFIG.get("smoothing_window", 25))
    plot_scaling_law_vs_N(summary_df, scaling_plot_path)
    plot_final_bars(summary_df, bars_plot_path)

    print("\n" + "=" * 50)
    print("Final summary table")
    print(summary_df.to_string(index=False))
    print("\nSaved files:")
    print(f"  - {summary_path}")
    print(f"  - {curves_path}")
    print(f"  - {curves_plot_path}")
    print(f"  - {scaling_plot_path}")
    print(f"  - {bars_plot_path}")
    print(f"  - {script_config_path}")


if __name__ == "__main__":
    main()
