"""Regenerate figures_v3/atrpa_N_scaling_v3.png using the deterministic
mean-field exploitability metric (no Monte-Carlo sampling noise) so the
TRPA O(N^{-1/4}) bias floor is visible."""
import os, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm
from smfg_envs import K, ENV_MAKERS
from algorithms import run_atrpa, run_trpa, ENV_SEED, CI_Z, apply_log_format

OUTDIR = "figures_v3"; os.makedirs(OUTDIR, exist_ok=True)
plt.rcParams.update({"figure.dpi": 160, "axes.grid": True, "grid.alpha": 0.3,
                     "font.size": 10, "savefig.dpi": 160})

T = 2000; N_LIST = (20, 50, 100, 200, 500, 1000); SEEDS = list(range(10))
SIGMA = 0.0


def policy_bias(q, F):
    """max_a F(q)_a - q . F(q): deterministic mean-field exploitability."""
    fv = F(q); return float(fv.max() - q @ fv)


fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for col, (env_name, env_maker) in enumerate(ENV_MAKERS.items()):
    F = env_maker(K, ENV_SEED)
    last_M, last_S = [], []; avg_M, avg_S = [], []; trpa_M, trpa_S = [], []
    for N in tqdm(N_LIST, desc=env_name, leave=False):
        last_v, avg_v, trpa_v = [], [], []
        for s in SEEDS:
            a = run_atrpa(F, N, T, seed=s); t = run_trpa(F, N, T, seed=s)
            last_v.append(policy_bias(a["mu_last"], F))
            avg_v .append(policy_bias(a["mu_avg"],  F))
            trpa_v.append(policy_bias(t["mu_last"], F))
        for src, m, sd in [(last_v,last_M,last_S),(avg_v,avg_M,avg_S),(trpa_v,trpa_M,trpa_S)]:
            v = np.array(src); m.append(v.mean()); sd.append(v.std(ddof=1))
    last_M=np.array(last_M);avg_M=np.array(avg_M);trpa_M=np.array(trpa_M)
    last_S=np.array(last_S);avg_S=np.array(avg_S);trpa_S=np.array(trpa_S)
    N_arr = np.array(N_LIST, dtype=float)
    sl=np.polyfit(np.log(N_arr),np.log(last_M),1)[0]
    sa=np.polyfit(np.log(N_arr),np.log(avg_M), 1)[0]
    st=np.polyfit(np.log(N_arr),np.log(trpa_M),1)[0]
    ax=axes[col]
    ci = CI_Z/np.sqrt(len(SEEDS))
    ax.errorbar(N_arr, trpa_M, yerr=ci*trpa_S, marker="o", color="C0", lw=1.8,
                capsize=3, label=f"TRPA-Full last  ({st:+.2f})")
    ax.errorbar(N_arr, last_M, yerr=ci*last_S, marker="o", color="C1", lw=1.8,
                capsize=3, label=f"A-TRPA-Last     ({sl:+.2f})")
    ax.errorbar(N_arr, avg_M,  yerr=ci*avg_S,  marker="s", color="C4", lw=1.8,
                ls="--", capsize=3, label=fr"A-TRPA-Avg ({sa:+.2f})")
    ax.plot(N_arr, trpa_M[0]*(N_arr/N_arr[0])**(-1/4),
            "--", color="green", lw=1.4, alpha=0.6, label=r"$N^{-1/4}$ ref")
    ax.plot(N_arr, avg_M[0]*(N_arr/N_arr[0])**(-1/2),
            ":",  color="red",   lw=1.4, alpha=0.6, label=r"$N^{-1/2}$ ref")
    ax.set_xscale("log"); ax.set_yscale("log"); apply_log_format(ax,axis="x")
    ax.set_xticks(N_arr)
    ax.set_xlabel("$N$"); ax.set_ylabel(r"Mean-field exploitability $\max_a F(\bar\mu)_a - \bar\mu^\top F(\bar\mu)$")
    ax.set_title(f"{env_name} | $T$={T}")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(True, which="both", alpha=0.25)
    print(f"{env_name}: TRPA {st:+.3f}  Avg {sa:+.3f}  Last {sl:+.3f}")

fig.suptitle(rf"Theory-aligned $N$-scaling, mean-field exploitability metric "
             rf"(deterministic; no Monte-Carlo sampling noise). "
             rf"$T$={T}, $\sigma{{=}}{SIGMA}$, 95\% CI over {len(SEEDS)} seeds.",
             y=1.02)
plt.tight_layout()
path = os.path.join(OUTDIR, "atrpa_N_scaling_v3.png")
plt.savefig(path, bbox_inches="tight"); plt.close()
print(f"[OK] saved {path}")
