"""
A-TRPA experiments for static mean-field games (SMFG).

Reproduces:
  - smfg_learning_curves_repro.png    (Fig 1: 2x3 panel, TRPA vs A-TRPA on 3 envs)
  - smfg_scaling_vs_N_repro.png       (Fig 2: 1x3 panel, scaling vs N)
  - smfg_baselines.png          (Fig 3: baseline comparison with OMD and FP)

Run:
    python smfg_experiments.py

Dependencies:  numpy, matplotlib, tqdm
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


# ============================================================
# 1. Environments — payoff operators F: Δ_K -> R^K
# ============================================================

def make_linear_env(K, seed):
    """F(mu) = -S mu + b with S random PSD (monotone, Lipschitz)."""
    rng = np.random.RandomState(seed)
    A = rng.randn(K, K) / np.sqrt(K)
    S = A.T @ A + 0.1 * np.eye(K)            # PSD
    b = rng.rand(K)

    def F(mu):
        return -S @ mu + b
    return F


def make_kl_env(K, seed):
    """KL-style monotone payoff: F_a(mu) = -log(mu(a) + eps) (decreasing in mu(a))."""
    rng = np.random.RandomState(seed)
    target = rng.rand(K); target /= target.sum()
    eps = 1e-3

    def F(mu):
        return -np.log(mu + eps) + np.log(target + eps)
    return F


def make_bb_env(K, seed):
    """Beach-Bar (stateless): F_a(mu) = 1 - |mu(a) - target(a)|."""
    rng = np.random.RandomState(seed)
    targets = np.linspace(0.1, 0.9, K)        # K beach locations
    rng.shuffle(targets)

    def F(mu):
        return 1.0 - np.abs(mu - targets)
    return F


ENV_MAKERS = {"Linear": make_linear_env, "KL": make_kl_env, "BB": make_bb_env}


# ============================================================
# 2. Utilities
# ============================================================

def project_simplex_batch(X):
    """
    Project each row of X (shape N×K) onto probability simplex.
    Sort-based algorithm (Duchi et al. 2008), vectorized over rows.
    """
    N, K = X.shape
    U = np.sort(X, axis=1)[:, ::-1]                       # descending
    cssv = np.cumsum(U, axis=1) - 1
    ind = np.arange(1, K + 1)
    cond = U - cssv / ind > 0
    rho = (cond * ind).max(axis=1)                        # 1..K
    theta = cssv[np.arange(N), rho - 1] / rho
    return np.maximum(X - theta[:, None], 0.0)


def sample_actions(policies, rng):
    """Sample one action per agent. policies: (N, K) → returns (N,)."""
    cumsum = np.cumsum(policies, axis=1)
    u = rng.rand(policies.shape[0], 1)
    return (cumsum > u).argmax(axis=1)


def empirical_distribution(actions, K):
    return np.bincount(actions, minlength=K) / len(actions)


def max_exploitability(policies, F):
    """
    Mean-field exploitability proxy:
        max_i [ max_a F(mu_bar)(a) - pi^i . F(mu_bar) ]
    where mu_bar = (1/N) sum_i pi^i.  Avoids the noisy empirical mu.
    Captures the limiting behaviour as N -> inf.
    """
    mu_bar = policies.mean(axis=0)
    fv = F(mu_bar)
    best = fv.max()
    vals = policies @ fv                                  # (N,)
    return np.max(best - vals)


def empirical_exploitability(pi_bar, F, N, n_eval, rng):
    """
    Finite-population (empirical) exploitability of a symmetric policy pi_bar:
        Expl_N(pi_bar) = E_{mu_hat ~ Multinomial(N, pi_bar)/N}
                        [ max_a F(mu_hat)_a  -  pi_bar . F(mu_hat) ].
    Sampling-based: draw N i.i.d. actions from pi_bar, form mu_hat, evaluate.
    Repeated n_eval times and averaged. Unlike max_exploitability (which uses
    F(mu_bar) and is therefore N-insensitive once mu_bar has converged), this
    metric retains the O(N^{-1/2}) finite-population noise that distinguishes
    TRPA's N^{-1/4} bias floor from A-TRPA's N^{-1/2} rate.
    """
    K_dim = len(pi_bar)
    pi_safe = np.clip(pi_bar, 1e-12, None)
    pi_safe /= pi_safe.sum()
    total = 0.0
    for _ in range(n_eval):
        actions = rng.choice(K_dim, size=N, p=pi_safe)
        mu_hat = np.bincount(actions, minlength=K_dim) / N
        fv = F(mu_hat)
        total += fv.max() - pi_bar @ fv
    return total / n_eval


# ============================================================
# 3. Algorithms
# ============================================================

def run_trpa(F, N, K, T, tau_const, seed, sigma=0.1):
    """TRPA-Full of Yardim et al. (2025) with constant tau."""
    rng = np.random.RandomState(seed)
    pi = np.full((N, K), 1.0 / K)
    expls = np.empty(T)

    for t in range(T):
        actions = sample_actions(pi, rng)
        mu_hat = empirical_distribution(actions, K)
        rewards = F(mu_hat) + sigma * rng.randn(N, K)     # (N, K)
        eta_t = 1.0 / (tau_const * (t + 2))
        pi = project_simplex_batch((1 - eta_t * tau_const) * pi + eta_t * rewards)
        expls[t] = max_exploitability(pi, F)
    return expls


def run_atrpa(F, N, K, T, alpha, beta, seed, sigma=0.1):
    """A-TRPA: tau_t = t^{-alpha}, eta_t = t^{-beta}, with alpha+beta=1, beta>alpha."""
    rng = np.random.RandomState(seed)
    pi = np.full((N, K), 1.0 / K)
    expls = np.empty(T)

    for t in range(T):
        actions = sample_actions(pi, rng)
        mu_hat = empirical_distribution(actions, K)
        rewards = F(mu_hat) + sigma * rng.randn(N, K)
        s = max(t, 1)
        tau_t = s ** (-alpha)
        eta_t = s ** (-beta)
        pi = project_simplex_batch((1 - eta_t * tau_t) * pi + eta_t * rewards)
        expls[t] = max_exploitability(pi, F)
    return expls


def run_omd(F, N, K, T, seed, sigma=0.1):
    """Online Mirror Descent (entropic / Hedge), no Tikhonov regularization."""
    rng = np.random.RandomState(seed)
    log_pi = np.full((N, K), 0.0)                          # log-policy (unnormalized)
    pi = np.full((N, K), 1.0 / K)
    expls = np.empty(T)

    for t in range(T):
        actions = sample_actions(pi, rng)
        mu_hat = empirical_distribution(actions, K)
        rewards = F(mu_hat) + sigma * rng.randn(N, K)
        eta_t = 1.0 / np.sqrt(t + 1)
        log_pi = log_pi + eta_t * rewards
        log_pi -= log_pi.max(axis=1, keepdims=True)        # numerical stability
        pi = np.exp(log_pi)
        pi /= pi.sum(axis=1, keepdims=True)
        expls[t] = max_exploitability(pi, F)
    return expls


def run_fictitious_play(F, N, K, T, seed, sigma=0.1):
    """
    Fictitious Play: each agent best-responds to the running average mu_hat.
    Mixed strategy = empirical fraction of historical best-responses.
    """
    rng = np.random.RandomState(seed)
    pi = np.full((N, K), 1.0 / K)
    br_count = np.zeros((N, K))                           # historical BR frequencies
    mu_avg = np.full(K, 1.0 / K)
    expls = np.empty(T)

    for t in range(T):
        actions = sample_actions(pi, rng)
        mu_hat = empirical_distribution(actions, K)
        mu_avg = (mu_avg * t + mu_hat) / (t + 1)
        rewards = F(mu_avg) + sigma * rng.randn(N, K)
        br = rewards.argmax(axis=1)
        br_count[np.arange(N), br] += 1
        pi = br_count / br_count.sum(axis=1, keepdims=True)
        expls[t] = max_exploitability(pi, F)
    return expls


# ============================================================
# 4. Experiment runners
# ============================================================

K = 5
T = 1000
N_LIST = [20, 50, 100, 200, 500, 1000]
SEEDS = list(range(5))
SIGMA = 0.1
ATRPA_ALPHA, ATRPA_BETA = 1 / 3, 2 / 3
ENV_SEED = 42                                              # fixed env across runs


def run_with_seeds(algo, F, N, **kwargs):
    """Average over SEEDS, return (mean, std) over learning curves."""
    out = []
    for seed in SEEDS:
        expls = algo(F, N, K, T, seed=seed, sigma=SIGMA, **kwargs)
        out.append(expls)
    out = np.array(out)
    return out.mean(axis=0), out.std(axis=0)


# ------------------------------------------------------------
# Figure 1 — learning curves: 2 algos × 3 envs, varying N
# ------------------------------------------------------------
def plot_learning_curves():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(N_LIST)))

    for col, (env_name, env_maker) in enumerate(ENV_MAKERS.items()):
        F = env_maker(K, seed=ENV_SEED)

        # Top row: TRPA
        ax = axes[0, col]
        for n_idx, N in enumerate(tqdm(N_LIST, desc=f"TRPA {env_name}")):
            tau = N ** (-1 / 4)
            mean, std = run_with_seeds(run_trpa, F, N, tau_const=tau)
            t_axis = np.arange(1, T + 1)
            ax.plot(t_axis, mean, color=cmap[n_idx], label=f"N={N}")
            ax.fill_between(t_axis, mean - std, mean + std,
                            color=cmap[n_idx], alpha=0.2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{env_name} | TRPA-Full")
        if col == 0:
            ax.set_ylabel("Maximum exploitability")
            ax.legend(fontsize=8, loc="upper right")

        # Bottom row: A-TRPA
        ax = axes[1, col]
        for n_idx, N in enumerate(tqdm(N_LIST, desc=f"A-TRPA {env_name}")):
            mean, std = run_with_seeds(run_atrpa, F, N,
                                       alpha=ATRPA_ALPHA, beta=ATRPA_BETA)
            ax.plot(t_axis, mean, color=cmap[n_idx], label=f"N={N}")
            ax.fill_between(t_axis, mean - std, mean + std,
                            color=cmap[n_idx], alpha=0.2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{env_name} | A-TRPA-Full")
        ax.set_xlabel("Time (log scale)")
        if col == 0:
            ax.set_ylabel("Maximum exploitability")

    plt.tight_layout()
    plt.savefig("smfg_learning_curves_repro.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] saved smfg_learning_curves_repro.png")


# ------------------------------------------------------------
# Figure 2 — scaling vs N: theory references + empirical
# ------------------------------------------------------------
def plot_scaling_vs_N():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for col, (env_name, env_maker) in enumerate(ENV_MAKERS.items()):
        F = env_maker(K, seed=ENV_SEED)
        ax = axes[col]

        trpa_finals = np.zeros((len(N_LIST), 2))
        atrpa_finals = np.zeros((len(N_LIST), 2))
        tail = T // 10                                    # average final 10% of T

        for n_idx, N in enumerate(tqdm(N_LIST, desc=f"Scaling {env_name}")):
            tau = N ** (-1 / 4)
            trpa_seeds = []; atrpa_seeds = []
            for seed in SEEDS:
                trpa_seeds.append(
                    run_trpa(F, N, K, T, tau, seed=seed, sigma=SIGMA)[-tail:].mean())
                atrpa_seeds.append(
                    run_atrpa(F, N, K, T, ATRPA_ALPHA, ATRPA_BETA,
                              seed=seed, sigma=SIGMA)[-tail:].mean())
            trpa_finals[n_idx] = (np.mean(trpa_seeds), np.std(trpa_seeds))
            atrpa_finals[n_idx] = (np.mean(atrpa_seeds), np.std(atrpa_seeds))

        N_arr = np.array(N_LIST, dtype=float)
        anchor = trpa_finals[0, 0]                         # anchor refs at N=20
        ref_quarter = anchor * (N_arr / N_arr[0]) ** (-1 / 4)
        ref_half = anchor * (N_arr / N_arr[0]) ** (-1 / 2)

        ax.loglog(N_arr, ref_quarter, "--", color="green",
                  label=r"Reference $N^{-1/4}$")
        ax.loglog(N_arr, ref_half, ":", color="red",
                  label=r"Reference $N^{-1/2}$")
        ax.errorbar(N_arr, trpa_finals[:, 0], yerr=trpa_finals[:, 1],
                    marker="o", color="C0", label="TRPA-Full")
        ax.errorbar(N_arr, atrpa_finals[:, 0], yerr=atrpa_finals[:, 1],
                    marker="o", color="C1", label="A-TRPA-Full")
        ax.set_xlabel("Number of Agents (N)")
        ax.set_ylabel("Final Exploitability")
        ax.set_title(f"{env_name} - Scaling with N")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("smfg_scaling_vs_N_repro.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] saved smfg_scaling_vs_N_repro.png")


# ------------------------------------------------------------
# Figure 3 (NEW) — baselines: TRPA, A-TRPA, OMD, FP at fixed N
# ------------------------------------------------------------
def plot_baselines(N=200):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    algos = {
        "TRPA-Full":   ("C0", lambda F, seed: run_trpa(
                              F, N, K, T, N ** (-1 / 4), seed=seed, sigma=SIGMA)),
        "A-TRPA-Full": ("C1", lambda F, seed: run_atrpa(
                              F, N, K, T, ATRPA_ALPHA, ATRPA_BETA,
                              seed=seed, sigma=SIGMA)),
        "OMD (Hedge)": ("C2", lambda F, seed: run_omd(
                              F, N, K, T, seed=seed, sigma=SIGMA)),
        "Fict. Play":  ("C3", lambda F, seed: run_fictitious_play(
                              F, N, K, T, seed=seed, sigma=SIGMA)),
    }

    for col, (env_name, env_maker) in enumerate(ENV_MAKERS.items()):
        F = env_maker(K, seed=ENV_SEED)
        ax = axes[col]
        for name, (color, runner) in algos.items():
            curves = []
            for seed in tqdm(SEEDS, desc=f"{name} {env_name}", leave=False):
                curves.append(runner(F, seed))
            curves = np.array(curves)
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            t_axis = np.arange(1, T + 1)
            ax.plot(t_axis, mean, color=color, label=name)
            ax.fill_between(t_axis, mean - std, mean + std,
                            color=color, alpha=0.15)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{env_name} | N={N}")
        ax.set_xlabel("Time (log scale)")
        if col == 0:
            ax.set_ylabel("Maximum exploitability")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("smfg_baselines.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] saved smfg_baselines.png")


# ============================================================
# 5. Main
# ============================================================
if __name__ == "__main__":
    plot_learning_curves()
    plot_scaling_vs_N()
    plot_baselines(N=200)
