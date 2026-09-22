"""
regen_v3.py
Regenerate the theory-aligned figures into figures_v3/, matching the
statement of Theorem~\\ref{thm:main} (averaged iterate $\\bar\\pi_T^i$).

Produces:
  figures_v3/atrpa_N_scaling_v3.png
      N-scaling at fixed T with three curves per environment:
        TRPA-Full last,  A-TRPA last,  A-TRPA avg ($\\bar\\pi_T$).
      This is the canonical theory-aligned figure: A-TRPA-Avg is what
      Theorem~\\ref{thm:main} bounds; A-TRPA-Last is the practical output.

  figures_v3/learning_curves_v3.png
      2x3 learning curves, one panel per (algorithm, environment):
        TRPA-Full (top row), A-TRPA last + A-TRPA avg (bottom row,
        averaged shown as dashed). Six $N$ values per panel.

  figures_v3/atrpa_T_decay_v3.png
      T-decay at fixed N=200 across three environments, with three
      series per panel: TRPA-Full last, A-TRPA last, A-TRPA avg, and
      $T^{-1/2}$ reference.

Reuses run_atrpa/run_trpa from verify_convergence (which already
maintains both the last iterate $\\pi_T^i$ and the eta-weighted
average $\\bar\\pi_T^i = (\\sum_t \\eta_t \\pi_t^i)/(\\sum_t \\eta_t)$.

Run: python regen_v3.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from smfg_envs import (
    K, ENV_MAKERS,
    empirical_exploitability, max_exploitability,
)
from algorithms import (
    run_atrpa, run_trpa,
    rolling_mean, ci_band, apply_log_format,
    SIGMA, SMOOTH_WINDOW, CI_Z, ENV_SEED,
)

OUTDIR = "figures_v3"
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 160, "axes.grid": True, "grid.alpha": 0.3,
                     "font.size": 10, "savefig.dpi": 160})


# ============================================================
# Figure 1: N-scaling, theory-aligned (A-TRPA-Avg explicit)
# ============================================================

def fig_n_scaling_v3(T=2000,
                     N_LIST=(20, 50, 100, 200, 500, 1000),
                     n_eval=50, seeds=None):
    """N-scaling with TRPA-Full last, A-TRPA last, A-TRPA avg.
       Empirical homogeneous-deployment exploitability metric:
         Expl_N(q) = E_{\\hat\\mu \\sim q^{\\otimes N}}
                       [ max_a F(\\hat\\mu)_a - q . F(\\hat\\mu) ].
       q is the population average of policies (mu_last for last
       iterate, mu_avg for averaged iterate)."""
    if seeds is None:
        seeds = list(range(10))
    n_seeds = len(seeds)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    summary = {}

    for col, (env_name, env_maker) in enumerate(ENV_MAKERS.items()):
        F = env_maker(K, ENV_SEED)
        last_finals, avg_finals, trpa_finals = [], [], []
        for N in tqdm(N_LIST, desc=f"N-scaling {env_name}", leave=False):
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
                m = arr_np.mean()
                sd = arr_np.std(ddof=1) if n_seeds > 1 else 0.0
                ci = CI_Z * sd / np.sqrt(n_seeds)
                target.append((m, ci, sd))

        last_finals = np.array(last_finals)
        avg_finals = np.array(avg_finals)
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
        ax.errorbar(N_arr, avg_finals[:, 0], yerr=avg_finals[:, 1],
                    marker="s", color="C4", lw=1.8, ls="--", capsize=3,
                    label=fr"A-TRPA avg ($\bar\pi_T$, theory) ({slope_avg:+.2f})")
        ax.plot(N_arr, trpa_finals[0, 0] * (N_arr / N_arr[0]) ** (-1/4),
                "--", color="green", lw=1.4, alpha=0.6, label=r"$N^{-1/4}$ ref")
        ax.plot(N_arr, avg_finals[0, 0] * (N_arr / N_arr[0]) ** (-1/2),
                ":", color="red", lw=1.4, alpha=0.6, label=r"$N^{-1/2}$ ref")

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
        }

    fig.suptitle(rf"Theory-aligned $N$-scaling: TRPA-Full last vs.\ A-TRPA "
                 rf"last vs.\ A-TRPA avg ($\bar\pi_T$, the iterate Theorem"
                 rf"~\ref{{thm:main}} bounds). $T$={T}, $\sigma{{=}}{SIGMA}$, "
                 rf"95\% CI over {n_seeds} seeds.", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTDIR, "atrpa_N_scaling_v3.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {path}")
    return summary


# ============================================================
# Figure 2: Learning curves with A-TRPA-Avg overlay
# ============================================================

def fig_learning_curves_v3(T=2000,
                            N_LIST=(20, 50, 100, 200, 500, 1000),
                            seeds=None):
    """2x3 grid: TRPA-Full last (top), A-TRPA last solid + A-TRPA avg
       dashed (bottom). Each panel overlays 6 N values."""
    if seeds is None:
        seeds = list(range(10))
    n_seeds = len(seeds)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5), sharex=True)
    cmap = plt.cm.viridis(np.linspace(0.05, 0.95, len(N_LIST)))
    t_axis = np.arange(1, T + 1)

    for col, (env_name, env_maker) in enumerate(ENV_MAKERS.items()):
        F = env_maker(K, ENV_SEED)

        # -------- top: TRPA-Full last --------
        ax = axes[0, col]
        for n_idx, N in enumerate(tqdm(N_LIST, desc=f"TRPA {env_name}",
                                       leave=False)):
            curves = np.array([
                run_trpa(F, N, T, seed=s, track_traj=True)["last_expl_traj"]
                for s in seeds
            ])
            mean, lo, hi = ci_band(curves)
            mean_s = rolling_mean(np.maximum(mean, 1e-12))
            lo_s = rolling_mean(np.maximum(lo, 1e-12))
            hi_s = rolling_mean(np.maximum(hi, 1e-12))
            ax.plot(t_axis, mean_s, color=cmap[n_idx], lw=1.5, label=f"N={N}")
            ax.fill_between(t_axis, lo_s, hi_s, color=cmap[n_idx],
                            alpha=0.18, lw=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{env_name} | TRPA-Full (last)")
        ax.grid(True, which="both", alpha=0.25)
        if col == 0:
            ax.set_ylabel("Max exploitability")
            ax.legend(fontsize=8, loc="upper right")

        # -------- bottom: A-TRPA last (solid) + A-TRPA avg (dashed) --------
        ax = axes[1, col]
        for n_idx, N in enumerate(tqdm(N_LIST, desc=f"A-TRPA {env_name}",
                                       leave=False)):
            results = [run_atrpa(F, N, T, seed=s, track_traj=True)
                       for s in seeds]
            last_curves = np.array([r["last_expl_traj"] for r in results])
            avg_curves = np.array([r["avg_expl_traj"] for r in results])

            l_mean, l_lo, l_hi = ci_band(last_curves)
            l_mean_s = rolling_mean(np.maximum(l_mean, 1e-12))
            l_lo_s = rolling_mean(np.maximum(l_lo, 1e-12))
            l_hi_s = rolling_mean(np.maximum(l_hi, 1e-12))
            ax.plot(t_axis, l_mean_s, color=cmap[n_idx], lw=1.5,
                    label=f"N={N} last")
            ax.fill_between(t_axis, l_lo_s, l_hi_s, color=cmap[n_idx],
                            alpha=0.10, lw=0)

            a_mean, _, _ = ci_band(avg_curves)
            a_mean_s = rolling_mean(np.maximum(a_mean, 1e-12))
            ax.plot(t_axis, a_mean_s, color=cmap[n_idx], lw=1.4, ls="--",
                    alpha=0.85)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{env_name} | A-TRPA: last (solid), "
                     fr"avg $\bar\pi_T$ (dashed)")
        ax.set_xlabel("Time (log scale)")
        ax.grid(True, which="both", alpha=0.25)
        if col == 0:
            ax.set_ylabel("Max exploitability")

    fig.suptitle(rf"Learning curves. Top: TRPA-Full last iterate. "
                 rf"Bottom: A-TRPA last iterate (solid) and weighted "
                 rf"averaged iterate $\bar\pi_T^i$ (dashed; the iterate "
                 rf"Theorem~\ref{{thm:main}} bounds). "
                 rf"$\sigma{{=}}{SIGMA}$, 95\% CI over {n_seeds} seeds, "
                 rf"smoothed (window={SMOOTH_WINDOW}).", y=1.00)
    plt.tight_layout()
    path = os.path.join(OUTDIR, "learning_curves_v3.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {path}")


# ============================================================
# Figure 3: T-decay with A-TRPA-Avg
# ============================================================

def fig_t_decay_v3(T=4000, N=200, seeds=None):
    if seeds is None:
        seeds = list(range(10))
    n_seeds = len(seeds)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for col, (env_name, env_maker) in enumerate(ENV_MAKERS.items()):
        F = env_maker(K, ENV_SEED)
        atrpa_last_curves, atrpa_avg_curves, trpa_curves = [], [], []
        for s in tqdm(seeds, desc=f"T-decay {env_name}", leave=False):
            a = run_atrpa(F, N, T, seed=s, track_traj=True)
            atrpa_last_curves.append(a["last_expl_traj"])
            atrpa_avg_curves.append(a["avg_expl_traj"])
            trpa_curves.append(run_trpa(F, N, T, seed=s,
                                        track_traj=True)["last_expl_traj"])
        atrpa_last_curves = np.array(atrpa_last_curves)
        atrpa_avg_curves = np.array(atrpa_avg_curves)
        trpa_curves = np.array(trpa_curves)

        l_mean, l_lo, l_hi = ci_band(atrpa_last_curves)
        a_mean, a_lo, a_hi = ci_band(atrpa_avg_curves)
        t_mean, t_lo, t_hi = ci_band(trpa_curves)

        l_mean_s = rolling_mean(l_mean, SMOOTH_WINDOW)
        l_lo_s = rolling_mean(np.maximum(l_lo, 1e-12), SMOOTH_WINDOW)
        l_hi_s = rolling_mean(l_hi, SMOOTH_WINDOW)
        a_mean_s = rolling_mean(a_mean, SMOOTH_WINDOW)
        a_lo_s = rolling_mean(np.maximum(a_lo, 1e-12), SMOOTH_WINDOW)
        a_hi_s = rolling_mean(a_hi, SMOOTH_WINDOW)
        t_mean_s = rolling_mean(t_mean, SMOOTH_WINDOW)
        t_lo_s = rolling_mean(np.maximum(t_lo, 1e-12), SMOOTH_WINDOW)
        t_hi_s = rolling_mean(t_hi, SMOOTH_WINDOW)

        ax = axes[col]
        t_ax = np.arange(1, T + 1)
        ax.fill_between(t_ax, t_lo_s, t_hi_s, color="C0", alpha=0.15, lw=0)
        ax.plot(t_ax, t_mean_s, color="C0", lw=1.6,
                label=r"TRPA-Full last ($\tau{=}N^{-1/4}$)")
        ax.fill_between(t_ax, l_lo_s, l_hi_s, color="C1", alpha=0.15, lw=0)
        ax.plot(t_ax, l_mean_s, color="C1", lw=1.6, label="A-TRPA last")
        ax.fill_between(t_ax, a_lo_s, a_hi_s, color="C4", alpha=0.15, lw=0)
        ax.plot(t_ax, a_mean_s, color="C4", lw=1.6, ls="--",
                label=r"A-TRPA avg $\bar\pi_T$ (theory)")

        warmup = max(1, T // 20)
        slope_l, _ = np.polyfit(np.log(t_ax[warmup:]),
                                np.log(np.maximum(l_mean_s[warmup:], 1e-12)), 1)
        slope_a, _ = np.polyfit(np.log(t_ax[warmup:]),
                                np.log(np.maximum(a_mean_s[warmup:], 1e-12)), 1)
        slope_t, _ = np.polyfit(np.log(t_ax[warmup:]),
                                np.log(np.maximum(t_mean_s[warmup:], 1e-12)), 1)
        anchor = a_mean_s[warmup]
        ax.plot(t_ax, anchor * (t_ax / t_ax[warmup]) ** (-0.5),
                ":", color="red", lw=1.2, alpha=0.7, label=r"$T^{-1/2}$ ref")

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("$T$"); ax.set_ylabel("Max exploitability")
        ax.set_title(f"{env_name} | $N$={N}\n"
                     f"slopes: avg {slope_a:+.2f}, last {slope_l:+.2f}, "
                     f"TRPA {slope_t:+.2f}")
        ax.legend(fontsize=7.5, loc="lower left")
        ax.grid(True, which="both", alpha=0.25)

    fig.suptitle(rf"Exploitability decay in $T$ ($\sigma$={SIGMA}, "
                 rf"95\% CI over {n_seeds} seeds)", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTDIR, "atrpa_T_decay_v3.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {path}")


if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv
    if quick:
        N_LIST = (20, 50, 100, 200)
        SEEDS = list(range(3))
        T_FAST = 500
        T_SCAL = 800
        print("(quick mode: 3 seeds, smaller N grid, T=500/800)")
    else:
        N_LIST = (20, 50, 100, 200, 500, 1000)
        SEEDS = list(range(10))
        T_FAST = 4000
        T_SCAL = 2000

    print("=" * 60)
    print(" Figure 1: N-scaling with A-TRPA-Avg (theory-aligned)")
    print("=" * 60)
    fig_n_scaling_v3(T=T_SCAL, N_LIST=N_LIST, seeds=SEEDS)

    print("\n" + "=" * 60)
    print(" Figure 2: Learning curves with A-TRPA-Avg overlay")
    print("=" * 60)
    fig_learning_curves_v3(T=T_SCAL, N_LIST=N_LIST, seeds=SEEDS)

    print("\n" + "=" * 60)
    print(" Figure 3: T-decay with A-TRPA-Avg")
    print("=" * 60)
    fig_t_decay_v3(T=T_FAST, N=200, seeds=SEEDS)

    print("\nAll v3 figures saved to:", os.path.abspath(OUTDIR))
