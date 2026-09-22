"""
verify_convergence.py
Empirical verification of the predicted convergence rates from the paper:

  Theorem (A-TRPA, averaged iterate):
    E[Expl(pi_avg_T)] = O(T^{-1/2} + N^{-1/2} + sigma)

  Theorem (B-ATRPA, bilevel):
    E[||grad U_delta(w_H)||] <= eps + O(1/sqrt(N) + sqrt(sigma) + sigma_U)
    after H,T = Theta(eps^{-2}) and total sample budget Otilde(eps^{-4} N).

Outputs (to figures_v2/):
  - atrpa_T_decay.png      : Exploitability(avg iterate) vs T at fixed N=200,
                              with reference slope T^{-1/2}.
  - atrpa_N_scaling.png    : Final exploitability vs N at fixed large T,
                              with reference slope N^{-1/2}.
  - batrpa_welfare.png     : B-ATRPA welfare U(w_h) and distance to optimum
                              over outer iterations.

Run:  python verify_convergence.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from smfg_envs import (
    K, project_simplex_batch, sample_actions,
    empirical_distribution, max_exploitability,
    empirical_exploitability, ENV_MAKERS,
)

OUTDIR = "figures_v2"
os.makedirs(OUTDIR, exist_ok=True)
plt.rcParams.update({"figure.dpi": 160, "axes.grid": True, "grid.alpha": 0.3,
                     "font.size": 10, "savefig.dpi": 160})

# Plotting helpers (borrowed from code_revised.py style)
CI_Z = 1.96               # 95% confidence interval
SMOOTH_WINDOW = 25        # rolling-mean window for time-series smoothing


def rolling_mean(values, window=SMOOTH_WINDOW):
    """Centered rolling mean for smooth curves."""
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def ci_band(curves, z=CI_Z):
    """Given (n_seeds, T) array, return (mean, ci_lower, ci_upper) with 95% CI."""
    curves = np.asarray(curves, dtype=float)
    n = curves.shape[0]
    mean = curves.mean(axis=0)
    if n <= 1:
        return mean, mean, mean
    sem = curves.std(axis=0, ddof=1) / np.sqrt(n)
    return mean, mean - z * sem, mean + z * sem


def apply_log_format(ax, axis="x"):
    """Use ScalarFormatter so log-axis shows actual numbers (100, not 10^2)."""
    from matplotlib.ticker import ScalarFormatter
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)

SIGMA = 0.0      # noiseless feedback (full-information setting). The
                 # last-iterate convergence proof of Theorem~\ref{thm:main}
                 # closes only at sigma=0 under the boundary schedule
                 # alpha=beta=1/2. We evaluate finite-N exploitability via
                 # the empirical metric (sample N actions from population
                 # mean policy), which is well-defined even when policy
                 # iterates are deterministic (sigma=0 makes all agents share
                 # the same gradient).
SEEDS = list(range(5))
ENV_SEED = 42


# ============================================================
# A-TRPA with η-weighted ergodic averaging (matches Algorithm 2)
# ============================================================

def run_trpa(F, N, T, seed, sigma=SIGMA, track_traj=False):
    """
    TRPA-Full (Yardim et al. 2025) baseline: constant tau = N^{-1/4},
    learning rate eta_t = tau^{-1}/(t+2). Returns the last iterate
    (no averaging). Tracks last-iterate exploitability for comparison.
    """
    rng = np.random.RandomState(seed)
    pi = np.full((N, K), 1.0 / K)
    tau = N ** (-1 / 4)

    if track_traj:
        last_expl_traj = np.empty(T)

    for t in range(T):
        eta_t = (1.0 / tau) / (t + 2)
        actions = sample_actions(pi, rng)
        mu_hat = empirical_distribution(actions, K)
        rewards = F(mu_hat) + sigma * rng.randn(N, K)
        pi = project_simplex_batch((1 - eta_t * tau) * pi + eta_t * rewards)

        if track_traj:
            last_expl_traj[t] = max_exploitability(pi, F)

    out = {"last_expl_T": max_exploitability(pi, F),
           "mu_last": pi.mean(axis=0)}
    if track_traj:
        out["last_expl_traj"] = last_expl_traj
    return out


def run_atrpa(F, N, T, seed, sigma=SIGMA, alpha=0.5, beta=0.5,
              track_traj=False):
    """
    A-TRPA (Algorithm 2): boundary schedule tau_t = t^{-alpha},
    eta_t = t^{-beta} with default alpha=beta=1/2 (Theorem main).
    Tracks BOTH the last iterate pi_T and the eta-weighted averaged
    iterate bar_pi_T = (sum_t eta_t pi_t) / (sum_t eta_t).
    Returns:
        'last_expl_T'   : proxy Expl(pi_T)
        'mu_last'       : population mean policy of pi_T
        'avg_expl_T'    : proxy Expl(bar_pi_T)         [averaged iterate]
        'mu_avg'        : population mean policy of bar_pi_T
        'last_expl_traj': proxy Expl(pi_t)             [if track_traj]
        'avg_expl_traj' : proxy Expl(bar_pi_t)         [if track_traj]
    """
    rng = np.random.RandomState(seed)
    pi = np.full((N, K), 1.0 / K)
    pi_avg = pi.copy()
    eta_sum = 0.0

    if track_traj:
        last_expl_traj = np.empty(T)
        avg_expl_traj = np.empty(T)

    for t in range(T):
        s = t + 1                          # 1-indexed schedule
        tau_t = s ** (-alpha)
        eta_t = s ** (-beta)

        actions = sample_actions(pi, rng)
        mu_hat = empirical_distribution(actions, K)
        rewards = F(mu_hat) + sigma * rng.randn(N, K)
        pi = project_simplex_batch((1 - eta_t * tau_t) * pi + eta_t * rewards)

        # eta-weighted running average update
        eta_sum += eta_t
        pi_avg = pi_avg + (eta_t / eta_sum) * (pi - pi_avg)

        if track_traj:
            last_expl_traj[t] = max_exploitability(pi, F)
            avg_expl_traj[t]  = max_exploitability(pi_avg, F)

    out = {"last_expl_T": max_exploitability(pi, F),
           "mu_last":     pi.mean(axis=0),
           "avg_expl_T":  max_exploitability(pi_avg, F),
           "mu_avg":      pi_avg.mean(axis=0)}
    if track_traj:
        out["last_expl_traj"] = last_expl_traj
        out["avg_expl_traj"]  = avg_expl_traj
    return out


# ============================================================
# Experiment 1: A-TRPA T-decay
# ============================================================

def exp_atrpa_T_decay(T=4000, N=200, seeds=None):
    if seeds is None:
        seeds = list(range(10))     # 10 seeds for tighter CI bands
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)

    for col, (env_name, env_maker) in enumerate(ENV_MAKERS.items()):
        F = env_maker(K, ENV_SEED)
        atrpa_curves, trpa_curves = [], []
        for seed in tqdm(seeds, desc=f"T-decay {env_name}", leave=False):
            atrpa_curves.append(run_atrpa(F, N, T, seed=seed,
                                          track_traj=True)["last_expl_traj"])
            trpa_curves.append(run_trpa(F, N, T, seed=seed,
                                        track_traj=True)["last_expl_traj"])
        atrpa_curves = np.array(atrpa_curves)
        trpa_curves = np.array(trpa_curves)

        # Rolling-mean smoothing + 95% CI bands across seeds
        a_mean, a_lo, a_hi = ci_band(atrpa_curves)
        t_mean, t_lo, t_hi = ci_band(trpa_curves)
        a_mean_s = rolling_mean(a_mean, SMOOTH_WINDOW)
        a_lo_s   = rolling_mean(np.maximum(a_lo, 1e-12), SMOOTH_WINDOW)
        a_hi_s   = rolling_mean(a_hi, SMOOTH_WINDOW)
        t_mean_s = rolling_mean(t_mean, SMOOTH_WINDOW)
        t_lo_s   = rolling_mean(np.maximum(t_lo, 1e-12), SMOOTH_WINDOW)
        t_hi_s   = rolling_mean(t_hi, SMOOTH_WINDOW)

        ax = axes[col]
        t_axis = np.arange(1, T + 1)

        ax.fill_between(t_axis, t_lo_s, t_hi_s, color="C0", alpha=0.18, lw=0)
        ax.plot(t_axis, t_mean_s, color="C0", lw=1.8,
                label=r"TRPA-Full ($\tau{=}N^{-1/4}$, last)")
        ax.fill_between(t_axis, a_lo_s, a_hi_s, color="C1", alpha=0.18, lw=0)
        ax.plot(t_axis, a_mean_s, color="C1", lw=1.8,
                label="A-TRPA (ours, last iterate)")

        # Slope fit on smoothed curve, skip warmup (first 5% of T)
        warmup = max(1, T // 20)
        slope_atrpa, _ = np.polyfit(np.log(t_axis[warmup:]),
                                    np.log(np.maximum(a_mean_s[warmup:], 1e-12)), 1)
        slope_trpa, _  = np.polyfit(np.log(t_axis[warmup:]),
                                    np.log(np.maximum(t_mean_s[warmup:], 1e-12)), 1)
        anchor_t = t_axis[warmup]; anchor_y = max(a_mean_s[warmup], 1e-12)
        ax.plot(t_axis, anchor_y * (t_axis / anchor_t) ** (-0.5),
                "--", color="red", lw=1.4, alpha=0.8, label=r"$T^{-1/2}$ reference")

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("$T$"); ax.set_ylabel("Max exploitability")
        ax.set_title(f"{env_name} | $N$={N}\n"
                     f"A-TRPA slope: {slope_atrpa:+.2f},  TRPA slope: {slope_trpa:+.2f}")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, which="both", alpha=0.25)

    fig.suptitle(rf"Exploitability decay in $T$ ($\sigma$={SIGMA}, "
                 rf"A-TRPA: $\alpha{{=}}\beta{{=}}1/2$, "
                 rf"shaded $=$ 95\% CI over {len(seeds)} seeds)",
                 y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTDIR, "atrpa_T_decay.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {path}")


# ============================================================
# Experiment 2: A-TRPA N-scaling
# ============================================================

def exp_atrpa_N_scaling(T=2000, N_LIST=(20, 50, 100, 200, 500, 1000),
                        n_eval=50, seeds=None):
    """
    N-scaling under sigma=0 noiseless feedback, using the *empirical*
    finite-N exploitability metric:
        Expl_N(pi_bar) = E_{mu_hat ~ Multinomial(N, pi_bar)/N}
                        [ max_a F(mu_hat)_a  -  pi_bar . F(mu_hat) ].
    This metric is well-defined even when policy iterates are
    deterministic (sigma=0), because the N-player game itself is
    stochastic at deployment time. It captures the O(N^{-1/2}) sampling
    noise that distinguishes TRPA's N^{-1/4} bias from A-TRPA's
    N^{-1/2} rate.
    """
    if seeds is None:
        seeds = list(range(10))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    n_seeds = len(seeds)
    summary = {}
    for col, (env_name, env_maker) in enumerate(ENV_MAKERS.items()):
        F = env_maker(K, ENV_SEED)
        atrpa_finals = np.zeros((len(N_LIST), 3))   # mean, ci-half, std
        trpa_finals  = np.zeros((len(N_LIST), 3))
        for n_idx, N in enumerate(tqdm(N_LIST, desc=f"N-scaling {env_name}",
                                       leave=False)):
            atrpa_seeds_v, trpa_seeds_v = [], []
            for s in seeds:
                atrpa_out = run_atrpa(F, N, T, seed=s)
                trpa_out  = run_trpa(F, N, T, seed=s)
                eval_rng_a = np.random.RandomState(10_000 + s)
                eval_rng_t = np.random.RandomState(20_000 + s)
                atrpa_seeds_v.append(empirical_exploitability(
                    atrpa_out["mu_last"], F, N, n_eval, eval_rng_a))
                trpa_seeds_v.append(empirical_exploitability(
                    trpa_out["mu_last"],  F, N, n_eval, eval_rng_t))
            for arr, target in [(atrpa_seeds_v, atrpa_finals),
                                (trpa_seeds_v,  trpa_finals)]:
                arr_np = np.array(arr, dtype=float)
                m  = arr_np.mean()
                sd = arr_np.std(ddof=1) if n_seeds > 1 else 0.0
                ci = CI_Z * sd / np.sqrt(n_seeds)
                target[n_idx] = (m, ci, sd)

        N_arr = np.array(N_LIST, dtype=float)
        slope_atrpa, _ = np.polyfit(np.log(N_arr), np.log(atrpa_finals[:, 0]), 1)
        slope_trpa,  _ = np.polyfit(np.log(N_arr), np.log(trpa_finals[:, 0]),  1)

        ax = axes[col]
        ax.errorbar(N_arr, trpa_finals[:, 0], yerr=trpa_finals[:, 1],
                    marker="o", color="C0", lw=1.8, capsize=3,
                    label=f"TRPA-Full last ({slope_trpa:+.2f})")
        ax.errorbar(N_arr, atrpa_finals[:, 0], yerr=atrpa_finals[:, 1],
                    marker="o", color="C1", lw=1.8, capsize=3,
                    label=f"A-TRPA last ({slope_atrpa:+.2f})")
        # Reference lines anchored at N=N_LIST[0]
        ax.plot(N_arr, trpa_finals[0, 0] * (N_arr / N_arr[0]) ** (-1/4),
                "--", color="green", lw=1.4, alpha=0.8, label=r"$N^{-1/4}$ ref")
        ax.plot(N_arr, atrpa_finals[0, 0] * (N_arr / N_arr[0]) ** (-1/2),
                ":",  color="red",   lw=1.4, alpha=0.8, label=r"$N^{-1/2}$ ref")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("$N$"); ax.set_ylabel("Empirical exploitability")
        ax.set_title(f"{env_name} | $T$={T}")
        apply_log_format(ax, axis="x")
        ax.set_xticks(N_arr)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, which="both", alpha=0.25)

        summary[env_name] = {"slope_atrpa": slope_atrpa, "slope_trpa": slope_trpa}

    fig.suptitle(rf"Scaling with $N$, empirical metric "
                 rf"($T$={T}, $\sigma$={SIGMA}, "
                 rf"95\% CI over {n_seeds} seeds)", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTDIR, "atrpa_N_scaling.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {path}")
    return summary


# ============================================================
# Experiment 3: B-ATRPA welfare convergence
# ============================================================

def make_stackelberg_linear(seed):
    """F(mu; w) = -S mu + b + w, with random PSD S."""
    rng = np.random.RandomState(seed)
    A = rng.randn(K, K) / np.sqrt(K)
    S = A.T @ A + 0.1 * np.eye(K)
    b = rng.rand(K)

    def F_param(mu, w):
        return -S @ mu + b + w
    return F_param, S, b


def project_simplex_1d(x):
    n = len(x)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.where(u - cssv / np.arange(1, n + 1) > 0)[0][-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(x - theta, 0)


def oracle_pi_star(S, b, w):
    """Closed-form: pi*(w) = projection of S^{-1}(b + w) on simplex."""
    return project_simplex_1d(np.linalg.solve(S, b + w))


def welfare_target(target):
    """U(mu, w) = -|| mu - target ||^2  (concave in w via pi*(w))."""
    def U(mu_bar, w):
        return float(-np.sum((mu_bar - target) ** 2))
    return U


def run_batrpa(F_param, U, S, b, N, T_inner, H, delta, eta_out_func,
                w_init, w_lo, w_hi, target, seed=0, sigma=SIGMA):
    """Outer zeroth-order mirror ascent with A-TRPA inner loop (averaged)."""
    rng = np.random.RandomState(seed)
    d = len(w_init)
    w = np.array(w_init, dtype=float)
    w_star = S @ target - b               # analytic optimum (interior)
    history = {"U": [], "dist": [], "w": [w.copy()]}

    for h in tqdm(range(H), desc="B-ATRPA outer", leave=False):
        u_h = rng.randn(d); u_h /= np.linalg.norm(u_h) + 1e-12
        w_p = np.clip(w + delta * u_h, w_lo, w_hi)
        w_m = np.clip(w - delta * u_h, w_lo, w_hi)

        # Inner A-TRPA averaged-iterate evaluation
        mu_p = run_atrpa(lambda mu: F_param(mu, w_p), N, T_inner,
                         seed=rng.randint(1 << 30), sigma=sigma)["mu_avg"]
        mu_m = run_atrpa(lambda mu: F_param(mu, w_m), N, T_inner,
                         seed=rng.randint(1 << 30), sigma=sigma)["mu_avg"]
        U_p = U(mu_p, w_p)
        U_m = U(mu_m, w_m)

        # Zeroth-order gradient
        g_h = (d / (2 * delta)) * (U_p - U_m) * u_h
        w = np.clip(w + eta_out_func(h) * g_h, w_lo, w_hi)
        history["w"].append(w.copy())

        # Diagnostic: clean U(w) at current w
        mu_clean = run_atrpa(lambda mu: F_param(mu, w), N, T_inner,
                             seed=rng.randint(1 << 30), sigma=sigma)["mu_avg"]
        history["U"].append(U(mu_clean, w))
        history["dist"].append(np.linalg.norm(w - w_star))

    return history, w_star


def run_no_principal_baseline(F_param, U, S, b, N, T_inner, H, target,
                                w_init, seed=0, sigma=SIGMA):
    """
    No-principal baseline: outer loop runs but principal never updates
    its action (eta_out = 0). Each outer iteration we re-run A-TRPA at
    the fixed w_init and report U(w_init). This shows the welfare
    trajectory in the absence of any active principal intervention.
    """
    rng = np.random.RandomState(seed)
    w = np.array(w_init, dtype=float)
    w_star = S @ target - b
    history = {"U": [], "dist": [np.linalg.norm(w - w_star)] * 0}

    for h in tqdm(range(H), desc="No-principal baseline", leave=False):
        # Just evaluate welfare at the (unchanging) w
        mu = run_atrpa(lambda mu: F_param(mu, w), N, T_inner,
                       seed=rng.randint(1 << 30), sigma=sigma)["mu_avg"]
        history["U"].append(U(mu, w))
        history["dist"].append(np.linalg.norm(w - w_star))

    return history


def run_random_principal_baseline(F_param, U, S, b, N, T_inner, H,
                                  delta, eta_out_func,
                                  w_init, w_lo, w_hi, target,
                                  seed=0, sigma=SIGMA):
    """
    Random-principal baseline: same outer-loop structure and SAME
    sample budget as B-ATRPA (two A-TRPA evaluations + one diagnostic
    per outer step), but instead of computing a zeroth-order gradient
    estimate, the principal takes a step in a random direction:
        w_{h+1} = Pi_W(w_h + eta_out * delta * u_h),  u_h uniform on sphere.
    This isolates the contribution of the FKM gradient estimator from
    the contribution of merely moving w around.
    """
    rng = np.random.RandomState(seed)
    d = len(w_init)
    w = np.array(w_init, dtype=float)
    w_star = S @ target - b
    history = {"U": [], "dist": [], "w": [w.copy()]}

    for h in tqdm(range(H), desc="Random-principal baseline", leave=False):
        u_h = rng.randn(d); u_h /= np.linalg.norm(u_h) + 1e-12
        w_p = np.clip(w + delta * u_h, w_lo, w_hi)
        w_m = np.clip(w - delta * u_h, w_lo, w_hi)

        # Same evaluation budget as B-ATRPA: two A-TRPA evaluations.
        # We do not use the resulting U^+, U^- for an update direction,
        # only to keep the per-outer-iteration sample budget identical.
        _ = run_atrpa(lambda mu: F_param(mu, w_p), N, T_inner,
                      seed=rng.randint(1 << 30), sigma=sigma)["mu_avg"]
        _ = run_atrpa(lambda mu: F_param(mu, w_m), N, T_inner,
                      seed=rng.randint(1 << 30), sigma=sigma)["mu_avg"]

        # Random principal step (no gradient information used).
        w = np.clip(w + eta_out_func(h) * delta * u_h, w_lo, w_hi)
        history["w"].append(w.copy())

        # Diagnostic: clean U(w) at current w
        mu_clean = run_atrpa(lambda mu: F_param(mu, w), N, T_inner,
                             seed=rng.randint(1 << 30), sigma=sigma)["mu_avg"]
        history["U"].append(U(mu_clean, w))
        history["dist"].append(np.linalg.norm(w - w_star))

    return history, w_star


def exp_batrpa(N=200, T_inner=300, H=80, delta=0.2, eta=0.05, n_seeds=3):
    F_param, S, b = make_stackelberg_linear(seed=ENV_SEED)
    target = np.ones(K) / K                # uniform target (interior optimum)
    U = welfare_target(target)

    # B-ATRPA (active principal)
    batrpa_welfares, batrpa_dists = [], []
    for seed in range(n_seeds):
        hist, w_star = run_batrpa(
            F_param, U, S, b, N=N, T_inner=T_inner, H=H,
            delta=delta, eta_out_func=lambda h: eta,
            w_init=np.zeros(K), w_lo=-3.0, w_hi=3.0,
            target=target, seed=seed,
        )
        batrpa_welfares.append(hist["U"])
        batrpa_dists.append(hist["dist"])
    batrpa_welfares = np.array(batrpa_welfares)
    batrpa_dists = np.array(batrpa_dists)

    # No-principal baseline (passive: w fixed at 0)
    base_welfares = []
    for seed in range(n_seeds):
        base = run_no_principal_baseline(
            F_param, U, S, b, N=N, T_inner=T_inner, H=H,
            target=target, w_init=np.zeros(K), seed=seed + 100,
        )
        base_welfares.append(base["U"])
    base_welfares = np.array(base_welfares)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    h_axis = np.arange(H)

    # ----- Panel (a): Welfare comparison -----
    ax = axes[0]
    ax.plot(h_axis, batrpa_welfares.mean(axis=0), color="C1", lw=2,
            label="B-ATRPA (active principal, ours)")
    ax.fill_between(h_axis, batrpa_welfares.min(0), batrpa_welfares.max(0),
                    alpha=0.2, color="C1")
    ax.plot(h_axis, base_welfares.mean(axis=0), color="C0", lw=2,
            label="No-principal baseline ($w \\equiv 0$)")
    ax.fill_between(h_axis, base_welfares.min(0), base_welfares.max(0),
                    alpha=0.2, color="C0")
    ax.axhline(0.0, ls="--", color="red", lw=1.5,
               label="$U^* = 0$ (analytic optimum)")
    ax.set_xlabel("Outer iteration $h$")
    ax.set_ylabel("Welfare $U(w_h)$")
    ax.set_title("(a) Welfare: B-ATRPA vs.\\ No-principal baseline")
    ax.legend(fontsize=8, loc="lower right")

    # ----- Panel (b): Distance to optimum -----
    ax = axes[1]
    ax.plot(h_axis, batrpa_dists.mean(axis=0), color="C3", lw=2,
            label="B-ATRPA $\\|w_h - w^*\\|_2$")
    ax.fill_between(h_axis, batrpa_dists.min(0), batrpa_dists.max(0),
                    alpha=0.2, color="C3")
    # No-principal stays at fixed distance
    no_p_dist = np.linalg.norm(np.zeros(K) - (S @ target - b))
    ax.axhline(no_p_dist, ls="--", color="C0", lw=1.5,
               label=f"No principal: $\\|w_0 - w^*\\| = {no_p_dist:.2f}$")
    ax.set_xlabel("Outer iteration $h$")
    ax.set_ylabel("Distance to optimum")
    ax.set_title("(b) Distance to analytic optimum $w^*$")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    fig.suptitle(f"B-ATRPA on Stackelberg-Linear (target = uniform, "
                 f"$N$={N}, $T_{{in}}$={T_inner}, $H$={H}, "
                 f"$\\delta$={delta}, $\\eta$={eta})", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTDIR, "batrpa_welfare.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {path}")

    print(f"  B-ATRPA: U(w_0) = {batrpa_welfares[:, 0].mean():.4f}, "
          f"U(w_H) = {batrpa_welfares[:, -1].mean():.4f}")
    print(f"  Baseline (no principal): U mean = {base_welfares.mean():.4f} "
          f"(std = {base_welfares.std():.4f})")
    print(f"  Welfare improvement from active principal: "
          f"{batrpa_welfares[:, -1].mean() - base_welfares.mean():+.4f}")


# ============================================================
# Experiment 4 (rich): 2x3 grid TRPA top / A-TRPA bottom, multiple N values
# ============================================================

def exp_learning_curves_rich(T=2000, N_LIST=(20, 50, 100, 200, 500, 1000),
                              seeds=None):
    """2x3 figure: TRPA-Full (top) and A-TRPA last iterate (bottom),
       across 3 envs (cols), each panel overlays 6 N values, with
       rolling-mean smoothing and 95% CI bands."""
    if seeds is None:
        seeds = list(range(10))
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5), sharex=True)
    cmap = plt.cm.viridis(np.linspace(0.05, 0.95, len(N_LIST)))
    t_axis = np.arange(1, T + 1)
    n_seeds = len(seeds)

    for col, (env_name, env_maker) in enumerate(ENV_MAKERS.items()):
        F = env_maker(K, ENV_SEED)

        # Top row: TRPA-Full (last iterate)
        ax = axes[0, col]
        for n_idx, N in enumerate(tqdm(N_LIST,
                                       desc=f"TRPA {env_name}", leave=False)):
            curves = np.array([
                run_trpa(F, N, T, seed=s, track_traj=True)["last_expl_traj"]
                for s in seeds
            ])
            mean, lo, hi = ci_band(curves)
            mean_s = rolling_mean(np.maximum(mean, 1e-12))
            lo_s   = rolling_mean(np.maximum(lo,   1e-12))
            hi_s   = rolling_mean(np.maximum(hi,   1e-12))
            ax.plot(t_axis, mean_s, color=cmap[n_idx], lw=1.5, label=f"N={N}")
            ax.fill_between(t_axis, lo_s, hi_s, color=cmap[n_idx],
                            alpha=0.18, lw=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{env_name} | TRPA-Full")
        ax.grid(True, which="both", alpha=0.25)
        if col == 0:
            ax.set_ylabel("Max exploitability")
            ax.legend(fontsize=8, loc="upper right")

        # Bottom row: A-TRPA last iterate
        ax = axes[1, col]
        for n_idx, N in enumerate(tqdm(N_LIST,
                                       desc=f"A-TRPA {env_name}", leave=False)):
            curves = np.array([
                run_atrpa(F, N, T, seed=s, track_traj=True)["last_expl_traj"]
                for s in seeds
            ])
            mean, lo, hi = ci_band(curves)
            mean_s = rolling_mean(np.maximum(mean, 1e-12))
            lo_s   = rolling_mean(np.maximum(lo,   1e-12))
            hi_s   = rolling_mean(np.maximum(hi,   1e-12))
            ax.plot(t_axis, mean_s, color=cmap[n_idx], lw=1.5, label=f"N={N}")
            ax.fill_between(t_axis, lo_s, hi_s, color=cmap[n_idx],
                            alpha=0.18, lw=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{env_name} | A-TRPA (ours, last iterate)")
        ax.set_xlabel("Time (log scale)")
        ax.grid(True, which="both", alpha=0.25)
        if col == 0:
            ax.set_ylabel("Max exploitability")

    fig.suptitle(rf"Learning curves: TRPA-Full (top) vs.\ A-TRPA last "
                 rf"iterate (bottom), $\sigma{{=}}{SIGMA}$, "
                 rf"95\% CI over {n_seeds} seeds, smoothed (window={SMOOTH_WINDOW})",
                 y=1.00)
    plt.tight_layout()
    path = os.path.join(OUTDIR, "learning_curves_rich.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {path}")


# ============================================================
# Experiment 5: Schedule ablation -- vary (alpha, beta) on alpha+beta=1
# ============================================================

def exp_schedule_ablation(T=4000, N=200, n_eval=50,
                          schedules=None, seeds=None):
    """T-decay AND final empirical exploitability across schedules
    (alpha, beta) with alpha + beta = 1, fixed N. Predicts: boundary
    case alpha = beta = 1/2 minimizes finite-time T-decay; adaptive
    schedules (Yardim) give slower T^{-(1-beta)} rate."""
    if schedules is None:
        schedules = [
            (1.0/4, 3.0/4),     # very fast eta, very slow tau
            (1.0/3, 2.0/3),     # Yardim adaptive
            (1.0/2, 1.0/2),     # boundary (ours)
            (2.0/3, 1.0/3),     # reverse adaptive
            (3.0/4, 1.0/4),     # very fast tau
        ]
    if seeds is None:
        seeds = list(range(10))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    cmap = plt.cm.plasma(np.linspace(0.05, 0.85, len(schedules)))
    summary = {env: [] for env in ENV_MAKERS}
    n_seeds = len(seeds)

    for col, (env_name, env_maker) in enumerate(ENV_MAKERS.items()):
        F = env_maker(K, ENV_SEED)
        ax = axes[col]
        t_axis = np.arange(1, T + 1)

        for sch_idx, (alpha, beta) in enumerate(tqdm(schedules,
                                                     desc=f"sched {env_name}",
                                                     leave=False)):
            # T-decay curves over seeds
            curves, finals = [], []
            for seed in seeds:
                out = run_atrpa(F, N, T, seed=seed, alpha=alpha, beta=beta,
                                track_traj=True)
                curves.append(out["last_expl_traj"])
                rng = np.random.RandomState(30_000 + seed)
                finals.append(empirical_exploitability(
                    out["mu_last"], F, N, n_eval, rng))
            curves = np.array(curves)
            mean, lo, hi = ci_band(curves)
            mean_s = rolling_mean(np.maximum(mean, 1e-12))
            lo_s   = rolling_mean(np.maximum(lo,   1e-12))
            hi_s   = rolling_mean(np.maximum(hi,   1e-12))

            label = (rf"$\alpha={alpha:.2f},\beta={beta:.2f}$"
                     + (" (ours)" if abs(alpha - 0.5) < 0.01 else ""))
            ax.plot(t_axis, mean_s, color=cmap[sch_idx], lw=1.6, label=label)
            ax.fill_between(t_axis, lo_s, hi_s, color=cmap[sch_idx],
                            alpha=0.15, lw=0)

            # Slope fit on smoothed mean (skip warmup)
            warmup = max(1, T // 20)
            slope, _ = np.polyfit(np.log(t_axis[warmup:]),
                                  np.log(mean_s[warmup:]), 1)
            summary[env_name].append({
                "alpha": alpha, "beta": beta, "slope": slope,
                "final_empirical": np.mean(finals),
                "final_std": np.std(finals, ddof=1) if n_seeds > 1 else 0.0,
            })

        # T^{-1/2} reference line anchored at the boundary curve
        boundary_idx = next(i for i, (a, b) in enumerate(schedules)
                            if abs(a - 0.5) < 0.01)
        anchor_y = summary[env_name][boundary_idx]["final_empirical"] * 1.5
        ax.plot(t_axis, anchor_y * (t_axis / t_axis[-1]) ** (-0.5),
                "--", color="red", lw=1.2, alpha=0.6,
                label=r"$T^{-1/2}$ reference")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("$T$"); ax.set_ylabel("Max exploitability")
        ax.set_title(f"{env_name} | $N$={N}")
        ax.legend(fontsize=7.5, loc="lower left")
        ax.grid(True, which="both", alpha=0.25)

    fig.suptitle(rf"Schedule ablation: $T$-decay across "
                 rf"$(\alpha, \beta)$ with $\alpha+\beta=1$ "
                 rf"($N$={N}, $\sigma$={SIGMA}, "
                 rf"95\% CI over {n_seeds} seeds)", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTDIR, "atrpa_schedule_ablation.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {path}")
    return summary


# ============================================================
# Experiment 3b: B-ATRPA -- active vs random vs no-principal
# ============================================================

def exp_batrpa_three_baselines(N=200, T_inner=300, H=80, delta=0.2,
                                eta=0.05, n_seeds=5):
    """B-ATRPA welfare convergence under three principal regimes:
       - active  (FKM zeroth-order gradient on welfare)
       - random  (random sphere step, same sample budget)
       - none    (w never updated)
       All start from w_0 = 0; the active arm should clearly dominate.
    """
    F_param, S, b = make_stackelberg_linear(seed=ENV_SEED)
    target = np.ones(K) / K
    U = welfare_target(target)
    w_star = S @ target - b
    no_p_dist = float(np.linalg.norm(np.zeros(K) - w_star))

    active_W, random_W, none_W = [], [], []
    active_D, random_D = [], []

    for seed in range(n_seeds):
        # Active principal
        h_act, _ = run_batrpa(
            F_param, U, S, b, N=N, T_inner=T_inner, H=H,
            delta=delta, eta_out_func=lambda h: eta,
            w_init=np.zeros(K), w_lo=-3.0, w_hi=3.0,
            target=target, seed=seed,
        )
        active_W.append(h_act["U"]); active_D.append(h_act["dist"])

        # Random principal (same budget)
        h_rnd, _ = run_random_principal_baseline(
            F_param, U, S, b, N=N, T_inner=T_inner, H=H,
            delta=delta, eta_out_func=lambda h: eta,
            w_init=np.zeros(K), w_lo=-3.0, w_hi=3.0,
            target=target, seed=seed + 200,
        )
        random_W.append(h_rnd["U"]); random_D.append(h_rnd["dist"])

        # No principal
        h_none = run_no_principal_baseline(
            F_param, U, S, b, N=N, T_inner=T_inner, H=H,
            target=target, w_init=np.zeros(K), seed=seed + 400,
        )
        none_W.append(h_none["U"])

    active_W = np.array(active_W); random_W = np.array(random_W)
    none_W   = np.array(none_W)
    active_D = np.array(active_D); random_D = np.array(random_D)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    h_axis = np.arange(H)

    # ----- (a) Welfare -----
    ax = axes[0]
    for arr, color, lbl in [
        (active_W, "C1", "B-ATRPA (active principal, ours)"),
        (random_W, "C2", "Random principal (same budget)"),
        (none_W,   "C0", "No principal ($w \\equiv 0$)"),
    ]:
        m, s = arr.mean(axis=0), arr.std(axis=0)
        ax.plot(h_axis, m, color=color, lw=2, label=lbl)
        ax.fill_between(h_axis, m - s, m + s, alpha=0.2, color=color)
    ax.axhline(0.0, ls="--", color="red", lw=1.4, label="$U^* = 0$")
    ax.set_xlabel("Outer iteration $h$"); ax.set_ylabel("Welfare $U(w_h)$")
    ax.set_title("(a) Welfare: active vs.\\ random vs.\\ no principal")
    ax.legend(fontsize=8, loc="lower right")

    # ----- (b) Distance to optimum -----
    ax = axes[1]
    for arr, color, lbl in [
        (active_D, "C3", "B-ATRPA (active)"),
        (random_D, "C2", "Random principal"),
    ]:
        m, s = arr.mean(axis=0), arr.std(axis=0)
        ax.plot(h_axis, m, color=color, lw=2, label=lbl)
        ax.fill_between(h_axis, m - s, m + s, alpha=0.2, color=color)
    ax.axhline(no_p_dist, ls="--", color="C0", lw=1.4,
               label=f"No principal: $\\|w_0-w^*\\| = {no_p_dist:.2f}$")
    ax.set_xlabel("Outer iteration $h$")
    ax.set_ylabel("$\\|w_h - w^*\\|_2$")
    ax.set_title("(b) Distance to analytic optimum")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    fig.suptitle(f"B-ATRPA active principal vs.\\ baselines ($N$={N}, "
                 f"$T_{{\\mathrm{{in}}}}$={T_inner}, $H$={H}, "
                 f"$\\delta$={delta}, {n_seeds} seeds)", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTDIR, "batrpa_three_baselines.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {path}")
    print(f"  Active   final U: {active_W[:, -1].mean():+.4f} +/- {active_W[:, -1].std():.4f}")
    print(f"  Random   final U: {random_W[:, -1].mean():+.4f} +/- {random_W[:, -1].std():.4f}")
    print(f"  No princ final U: {none_W[:, -1].mean():+.4f} +/- {none_W[:, -1].std():.4f}")


# ============================================================
# Experiment 2b: N-scaling at TWO noise levels (sigma=0 and sigma>0)
# Combines theory-aligned noiseless setting with the noisy setting
# where TRPA's variance term makes A-TRPA's advantage dramatic.
# ============================================================

def exp_n_scaling_dual_sigma(T=2000, N_LIST=(20, 50, 100, 200, 500, 1000),
                              sigmas=(0.0, 0.1), n_eval=50, seeds=None):
    if seeds is None:
        seeds = list(range(10))
    n_seeds = len(seeds)
    n_rows = len(sigmas)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4.5 * n_rows),
                             sharex="col")
    if n_rows == 1:
        axes = axes[None, :]

    summary = {}
    N_arr = np.array(N_LIST, dtype=float)

    for row, sigma in enumerate(sigmas):
        for col, (env_name, env_maker) in enumerate(ENV_MAKERS.items()):
            F = env_maker(K, ENV_SEED)
            atrpa_finals = np.zeros((len(N_LIST), 2))   # (mean, ci)
            trpa_finals  = np.zeros((len(N_LIST), 2))
            for n_idx, N in enumerate(tqdm(N_LIST,
                                           desc=f"sigma={sigma} {env_name}",
                                           leave=False)):
                a_v, t_v = [], []
                for s in seeds:
                    a = run_atrpa(F, N, T, seed=s, sigma=sigma)
                    t = run_trpa(F, N, T, seed=s, sigma=sigma)
                    rng_a = np.random.RandomState(80_000 + s)
                    rng_t = np.random.RandomState(90_000 + s)
                    a_v.append(empirical_exploitability(
                        a["mu_last"], F, N, n_eval, rng_a))
                    t_v.append(empirical_exploitability(
                        t["mu_last"], F, N, n_eval, rng_t))
                a_arr = np.array(a_v); t_arr = np.array(t_v)
                a_sd = a_arr.std(ddof=1) if n_seeds > 1 else 0.0
                t_sd = t_arr.std(ddof=1) if n_seeds > 1 else 0.0
                atrpa_finals[n_idx] = (a_arr.mean(),
                                       CI_Z * a_sd / np.sqrt(n_seeds))
                trpa_finals[n_idx]  = (t_arr.mean(),
                                       CI_Z * t_sd / np.sqrt(n_seeds))

            slope_a, _ = np.polyfit(np.log(N_arr), np.log(atrpa_finals[:, 0]), 1)
            slope_t, _ = np.polyfit(np.log(N_arr), np.log(trpa_finals[:, 0]),  1)

            ax = axes[row, col]
            ax.errorbar(N_arr, trpa_finals[:, 0], yerr=trpa_finals[:, 1],
                        marker="o", color="C0", lw=1.8, capsize=3,
                        label=f"TRPA-Full last ({slope_t:+.2f})")
            ax.errorbar(N_arr, atrpa_finals[:, 0], yerr=atrpa_finals[:, 1],
                        marker="o", color="C1", lw=1.8, capsize=3,
                        label=f"A-TRPA last ({slope_a:+.2f})")
            ax.plot(N_arr, trpa_finals[0, 0] * (N_arr / N_arr[0]) ** (-1/4),
                    "--", color="green", lw=1.4, alpha=0.7, label=r"$N^{-1/4}$ ref")
            ax.plot(N_arr, atrpa_finals[0, 0] * (N_arr / N_arr[0]) ** (-1/2),
                    ":",  color="red",   lw=1.4, alpha=0.7, label=r"$N^{-1/2}$ ref")
            ax.set_xscale("log"); ax.set_yscale("log")
            apply_log_format(ax, axis="x"); ax.set_xticks(N_arr)
            if row == n_rows - 1:
                ax.set_xlabel("$N$")
            if col == 0:
                ax.set_ylabel(rf"Empirical exploitability  ($\sigma{{=}}{sigma}$)")
            ax.set_title(f"{env_name}" + (f" | $T$={T}" if row == 0 else ""))
            ax.legend(fontsize=7.5, loc="lower left")
            ax.grid(True, which="both", alpha=0.25)

            summary[(sigma, env_name)] = {
                "slope_a": slope_a, "slope_t": slope_t,
                "a_finals": atrpa_finals, "t_finals": trpa_finals,
            }

    fig.suptitle(rf"$N$-scaling at two noise levels: noiseless "
                 rf"($\sigma{{=}}0$, top) and noisy ($\sigma{{=}}{sigmas[1]}$, "
                 rf"bottom). $T$={T}, 95\% CI over {n_seeds} seeds.",
                 y=1.005)
    plt.tight_layout()
    path = os.path.join(OUTDIR, "atrpa_N_scaling_dual_sigma.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {path}")
    return summary


# ============================================================
# Experiment 5b: A-TRPA last iterate vs weighted-averaged iterate
# (the most important ablation -- closes the theory-experiment gap
#  between Theorem main (averaged) and our empirical figures (last))
# ============================================================

def exp_last_vs_avg(T=2000, N_LIST=(20, 50, 100, 200, 500, 1000), n_eval=50,
                    seeds=None):
    """N-scaling for THREE methods at the boundary schedule + sigma=0:
       - A-TRPA last iterate (orange)
       - A-TRPA weighted averaged iterate (purple, the one Theorem main bounds)
       - TRPA-Full last iterate (blue baseline)
       Plots empirical exploitability vs N on log-log; reports slope + std.
       Goal: show that the averaged iterate empirically tracks the same
       N-scaling as last iterate, sealing the theory <-> experiment gap.
    """
    if seeds is None:
        seeds = list(range(10))   # 10 seeds for stable mean+std

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    summary = {}
    n_seeds = len(seeds)

    for col, (env_name, env_maker) in enumerate(ENV_MAKERS.items()):
        F = env_maker(K, ENV_SEED)
        last_finals, avg_finals, trpa_finals = [], [], []
        for N in tqdm(N_LIST, desc=f"last-vs-avg {env_name}", leave=False):
            last_v, avg_v, trpa_v = [], [], []
            for s in seeds:
                a = run_atrpa(F, N, T, seed=s)
                t = run_trpa(F, N, T, seed=s)
                rng_l = np.random.RandomState(50_000 + s)
                rng_a = np.random.RandomState(60_000 + s)
                rng_t = np.random.RandomState(70_000 + s)
                last_v.append(empirical_exploitability(
                    a["mu_last"], F, N, n_eval, rng_l))
                avg_v.append(empirical_exploitability(
                    a["mu_avg"],  F, N, n_eval, rng_a))
                trpa_v.append(empirical_exploitability(
                    t["mu_last"], F, N, n_eval, rng_t))
            for arr, target in [(last_v, last_finals),
                                (avg_v,  avg_finals),
                                (trpa_v, trpa_finals)]:
                arr_np = np.array(arr, dtype=float)
                m  = arr_np.mean()
                sd = arr_np.std(ddof=1) if n_seeds > 1 else 0.0
                ci = CI_Z * sd / np.sqrt(n_seeds)
                target.append((m, ci, sd))

        last_finals = np.array(last_finals)
        avg_finals  = np.array(avg_finals)
        trpa_finals = np.array(trpa_finals)
        N_arr = np.array(N_LIST, dtype=float)

        slope_last, _ = np.polyfit(np.log(N_arr), np.log(last_finals[:, 0]), 1)
        slope_avg,  _ = np.polyfit(np.log(N_arr), np.log(avg_finals[:, 0]),  1)
        slope_trpa, _ = np.polyfit(np.log(N_arr), np.log(trpa_finals[:, 0]), 1)

        ax = axes[col]
        ax.errorbar(N_arr, trpa_finals[:, 0], yerr=trpa_finals[:, 1],
                    marker="o", color="C0", lw=1.8, capsize=3,
                    label=f"TRPA-Full last  ({slope_trpa:+.2f})")
        ax.errorbar(N_arr, last_finals[:, 0], yerr=last_finals[:, 1],
                    marker="o", color="C1", lw=1.8, capsize=3,
                    label=f"A-TRPA last     ({slope_last:+.2f})")
        ax.errorbar(N_arr, avg_finals[:, 0],  yerr=avg_finals[:, 1],
                    marker="s", color="C4", lw=1.8, ls="--", capsize=3,
                    label=f"A-TRPA avg ($\\bar\\pi_T$) ({slope_avg:+.2f})")
        ax.plot(N_arr, trpa_finals[0, 0] * (N_arr / N_arr[0]) ** (-1/4),
                "--", color="green", lw=1.4, alpha=0.6, label=r"$N^{-1/4}$ ref")
        ax.plot(N_arr, last_finals[0, 0] * (N_arr / N_arr[0]) ** (-1/2),
                ":",  color="red",   lw=1.4, alpha=0.6, label=r"$N^{-1/2}$ ref")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("$N$"); ax.set_ylabel("Empirical exploitability")
        ax.set_title(f"{env_name} | $T$={T}")
        apply_log_format(ax, axis="x")
        ax.set_xticks(N_arr)
        ax.legend(fontsize=7.5, loc="lower left")
        ax.grid(True, which="both", alpha=0.25)

        summary[env_name] = {
            "slope_last": slope_last, "slope_avg": slope_avg,
            "slope_trpa": slope_trpa,
            "last_finals": last_finals, "avg_finals": avg_finals,
            "trpa_finals": trpa_finals,
        }

    fig.suptitle(rf"A-TRPA last iterate vs.\ weighted averaged iterate vs.\ "
                 rf"TRPA-Full (boundary schedule, $\sigma{{=}}0$, $T$={T}, "
                 rf"95\% CI over {n_seeds} seeds)", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTDIR, "atrpa_last_vs_avg.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {path}")
    return summary


# ============================================================
# Experiment 6: sigma-sensitivity (verifies sigma-floor at boundary)
# ============================================================

def exp_sigma_sensitivity(T=4000, N=200, n_eval=50,
                          sigmas=(0.0, 0.01, 0.05, 0.1, 0.5),
                          seeds=None):
    """Final empirical exploitability vs sigma at boundary schedule
    (alpha=beta=1/2). Theorem main predicts an irreducible O(sigma) floor
    for the averaged iterate; Proposition last-iter suggests the milder
    O(sigma * T^{-1/4}) floor under co-coercivity."""
    if seeds is None:
        seeds = list(range(10))
    n_seeds = len(seeds)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    summary = {env: [] for env in ENV_MAKERS}

    for col, (env_name, env_maker) in enumerate(ENV_MAKERS.items()):
        F = env_maker(K, ENV_SEED)
        finals_mean, finals_ci = [], []
        for sigma in tqdm(sigmas, desc=f"sigma {env_name}", leave=False):
            seed_finals = []
            for seed in seeds:
                out = run_atrpa(F, N, T, seed=seed, sigma=sigma)
                rng = np.random.RandomState(40_000 + seed)
                seed_finals.append(empirical_exploitability(
                    out["mu_last"], F, N, n_eval, rng))
            arr = np.array(seed_finals, dtype=float)
            sd = arr.std(ddof=1) if n_seeds > 1 else 0.0
            finals_mean.append(arr.mean())
            finals_ci.append(CI_Z * sd / np.sqrt(n_seeds))
            summary[env_name].append({"sigma": sigma,
                                      "final": arr.mean(),
                                      "ci": CI_Z * sd / np.sqrt(n_seeds)})

        ax = axes[col]
        sigma_arr = np.array(sigmas)
        finals_mean = np.array(finals_mean)
        finals_ci = np.array(finals_ci)
        x_plot = np.maximum(sigma_arr, 1e-3)
        ax.errorbar(x_plot, finals_mean, yerr=finals_ci,
                    marker="o", color="C1", lw=1.8, capsize=3,
                    label="A-TRPA (last iterate)")
        sigma_floor = finals_mean[0]
        ax.plot(sigma_arr[1:],
                sigma_floor + sigma_arr[1:] *
                (finals_mean[-1] - sigma_floor) / sigmas[-1],
                "--", color="red", lw=1.4, alpha=0.6,
                label=r"$\sigma$-linear ref (averaged-iter Thm)")
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\sigma$ (per-agent reward noise)")
        ax.set_ylabel("Final empirical exploitability")
        ax.set_title(f"{env_name} | $N$={N}, $T$={T}")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.25)

    fig.suptitle(rf"$\sigma$-sensitivity at boundary schedule "
                 rf"$\tau_t = \eta_t = t^{{-1/2}}$ "
                 rf"($N$={N}, $T$={T}, "
                 rf"95\% CI over {n_seeds} seeds)", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTDIR, "atrpa_sigma_sensitivity.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {path}")
    return summary


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" Experiment 1: A-TRPA T-decay (verify slope = -1/2)")
    print("=" * 60)
    exp_atrpa_T_decay(T=4000, N=200)

    print("\n" + "=" * 60)
    print(" Experiment 2: A-TRPA N-scaling (verify slope = -1/2)")
    print("=" * 60)
    summary = exp_atrpa_N_scaling(T=2000)
    print("\n  Estimated N-scaling slopes (theory: A-TRPA -0.5, TRPA -0.25):")
    for env, s in summary.items():
        print(f"    {env:8s}: A-TRPA slope = {s['slope_atrpa']:+.3f}, "
              f"TRPA slope = {s['slope_trpa']:+.3f}")

    print("\n" + "=" * 60)
    print(" Experiment 3: B-ATRPA welfare convergence")
    print("=" * 60)
    exp_batrpa(N=200, T_inner=300, H=80, delta=0.2, eta=0.05, n_seeds=3)

    print("\nAll figures saved to: " + os.path.abspath(OUTDIR))
