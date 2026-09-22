"""Regenerate §6.6 schedule ablation and §6.7 sigma sensitivity using
A-TRPA-Avg (the averaged iterate, primary output of A-TRPA).
Saves to figures_v3/.

Reduced budget: T=2000, 5 seeds, N=200."""

import os, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm
from smfg_envs import K, ENV_MAKERS, empirical_exploitability
from algorithms import (
    run_atrpa, rolling_mean, ci_band, apply_log_format,
    SMOOTH_WINDOW, CI_Z, ENV_SEED,
)

OUTDIR = "figures_v3"; os.makedirs(OUTDIR, exist_ok=True)
plt.rcParams.update({"figure.dpi": 160, "axes.grid": True, "grid.alpha": 0.3,
                     "font.size": 10, "savefig.dpi": 160})

T = 2000; N = 200; n_eval = 50; SEEDS = list(range(5))
SIGMA = 0.0


def fig_schedule_ablation_avg():
    schedules = [(1/4, 3/4), (1/3, 2/3), (1/2, 1/2), (2/3, 1/3), (3/4, 1/4)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    cmap = plt.cm.plasma(np.linspace(0.05, 0.85, len(schedules)))
    summary = {env: [] for env in ENV_MAKERS}

    for col, (env_name, env_maker) in enumerate(ENV_MAKERS.items()):
        F = env_maker(K, ENV_SEED)
        ax = axes[col]
        t_axis = np.arange(1, T + 1)
        for sch_idx, (alpha, beta) in enumerate(tqdm(schedules,
                                                     desc=f"sched {env_name}",
                                                     leave=False)):
            curves, finals = [], []
            for seed in SEEDS:
                out = run_atrpa(F, N, T, seed=seed, alpha=alpha, beta=beta,
                                track_traj=True)
                curves.append(out["avg_expl_traj"])      # <-- AVG
                rng = np.random.RandomState(30_000 + seed)
                finals.append(empirical_exploitability(
                    out["mu_avg"], F, N, n_eval, rng))   # <-- AVG
            curves = np.array(curves)
            mean, lo, hi = ci_band(curves)
            mean_s = rolling_mean(np.maximum(mean, 1e-12))
            lo_s = rolling_mean(np.maximum(lo, 1e-12))
            hi_s = rolling_mean(np.maximum(hi, 1e-12))
            label = (rf"$\alpha={alpha:.2f},\beta={beta:.2f}$"
                     + (" (ours)" if abs(alpha - 0.5) < 0.01 else ""))
            ax.plot(t_axis, mean_s, color=cmap[sch_idx], lw=1.6, label=label)
            ax.fill_between(t_axis, lo_s, hi_s, color=cmap[sch_idx],
                            alpha=0.15, lw=0)
            warmup = max(1, T // 20)
            slope, _ = np.polyfit(np.log(t_axis[warmup:]),
                                  np.log(mean_s[warmup:]), 1)
            summary[env_name].append({
                "alpha": alpha, "beta": beta, "slope": slope,
                "final_avg": float(np.mean(finals)),
                "final_std": float(np.std(finals, ddof=1)) if len(SEEDS) > 1 else 0.0,
            })
        boundary_idx = next(i for i, (a, b) in enumerate(schedules)
                            if abs(a - 0.5) < 0.01)
        anchor_y = summary[env_name][boundary_idx]["final_avg"] * 1.5
        ax.plot(t_axis, anchor_y * (t_axis / t_axis[-1]) ** (-0.5),
                "--", color="red", lw=1.2, alpha=0.6,
                label=r"$T^{-1/2}$ ref")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("$T$"); ax.set_ylabel(r"Avg-iterate exploitability $\mathrm{Expl}_N(\bar\pi^{\mathrm{pop}}_T)$")
        ax.set_title(f"{env_name} | $N$={N}")
        ax.legend(fontsize=7.5, loc="lower left")
        ax.grid(True, which="both", alpha=0.25)
    fig.suptitle(rf"Schedule ablation \emph{{(A-TRPA-Avg)}}: $T$-decay across "
                 rf"$(\alpha, \beta)$ with $\alpha+\beta=1$, "
                 rf"$N$={N}, $\sigma$={SIGMA}, "
                 rf"95\% CI over {len(SEEDS)} seeds.", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTDIR, "atrpa_schedule_ablation_avg.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"[OK] saved {path}")
    return summary


def fig_sigma_sensitivity_avg():
    sigmas = (0.0, 0.01, 0.05, 0.1, 0.5)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    summary = {env: [] for env in ENV_MAKERS}
    for col, (env_name, env_maker) in enumerate(ENV_MAKERS.items()):
        F = env_maker(K, ENV_SEED)
        finals_mean, finals_ci = [], []
        for sigma in tqdm(sigmas, desc=f"sigma {env_name}", leave=False):
            seed_finals = []
            for seed in SEEDS:
                out = run_atrpa(F, N, T, seed=seed, sigma=sigma)
                rng = np.random.RandomState(40_000 + seed)
                seed_finals.append(empirical_exploitability(
                    out["mu_avg"], F, N, n_eval, rng))    # <-- AVG
            arr = np.array(seed_finals, dtype=float)
            sd = arr.std(ddof=1) if len(SEEDS) > 1 else 0.0
            finals_mean.append(arr.mean())
            finals_ci.append(CI_Z * sd / np.sqrt(len(SEEDS)))
            summary[env_name].append({"sigma": sigma, "final": float(arr.mean()),
                                       "ci": float(CI_Z * sd / np.sqrt(len(SEEDS)))})
        ax = axes[col]
        sigma_arr = np.array(sigmas)
        finals_mean = np.array(finals_mean); finals_ci = np.array(finals_ci)
        x_plot = np.maximum(sigma_arr, 1e-3)
        ax.errorbar(x_plot, finals_mean, yerr=finals_ci,
                    marker="o", color="C4", lw=1.8, capsize=3,
                    label="A-TRPA-Avg")
        sigma_floor = finals_mean[0]
        ax.plot(sigma_arr[1:],
                sigma_floor + sigma_arr[1:] *
                (finals_mean[-1] - sigma_floor) / sigmas[-1],
                "--", color="red", lw=1.4, alpha=0.6,
                label=r"$\sigma$-linear ref")
        ax.set_xscale("symlog", linthresh=1e-3); ax.set_yscale("log")
        ax.set_xlabel(r"$\sigma$"); ax.set_ylabel("Final exploitability (avg iterate)")
        ax.set_title(f"{env_name} | $N$={N}, $T$={T}")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.25)
    fig.suptitle(rf"$\sigma$-sensitivity \emph{{(A-TRPA-Avg)}} at boundary "
                 rf"$\tau_t = \eta_t = t^{{-1/2}}$, "
                 rf"$N$={N}, $T$={T}, 95\% CI over {len(SEEDS)} seeds.",
                 y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTDIR, "atrpa_sigma_sensitivity_avg.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"[OK] saved {path}")
    return summary


if __name__ == "__main__":
    import json
    print("schedule ablation (avg) ...")
    s1 = fig_schedule_ablation_avg()
    print("\nsigma sensitivity (avg) ...")
    s2 = fig_sigma_sensitivity_avg()
    with open("v3_67_summary.json", "w") as f:
        json.dump({"schedule": s1, "sigma": s2}, f, indent=2)
    print("Saved v3_67_summary.json")
