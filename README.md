# Static Mean-Field Games: Annealed TRPA and Bilevel Intervention

A collaborative MVA course research project by **Jin Ying and Zhao Yutai**. The [report](Report_SMFG.pdf), titled *Dynamic Principal Intervention in Static Mean-Field Games: From Annealed TRPA to Bilevel Stackelberg Learning*, studies population learning and intervention in synthetic static mean-field games.

## Scope

The code compares a fixed-regularization TRPA baseline with an annealed variant and explores a bilevel principal-intervention extension. It includes linear, KL-style and beach-bar-style synthetic payoff environments, seeded experiments, last-iterate and averaged-iterate comparisons, and population/time scaling plots.

This is a **joint student research report**. The NeurIPS-style document template does not imply conference acceptance. The report's theoretical qualifications and unresolved last-iterate issues matter: the bilevel extension is exploratory, and empirical curves are not a general convergence guarantee. The material reviewed does not establish a file-by-file division of individual contributions; do not describe the entire project as a solo implementation.

## Contents and reproduction entry points

| File | Purpose / command |
|---|---|
| `smfg_envs.py` | Environments and baseline plots; `python smfg_envs.py` |
| `algorithms.py` | TRPA, annealed TRPA and bilevel experiments; `python algorithms.py` |
| `figures_main.py` | Main population/time scaling and trajectory figures; `python figures_main.py` |
| `figures_ablation.py` | Ablation plots; `python figures_ablation.py` |
| `code_revised.py` | Separate full-feedback comparison with finite-population Monte Carlo exploitability; `python code_revised.py` |
| `fix_n_scaling.py` | Additional population-scaling experiment |
| `figures_v2/`, `figures_v3/`, summary CSV/JSON | Selected saved outputs from the local project |

Install dependencies in an isolated Python environment:

```bash
python -m pip install -r requirements.txt
python figures_main.py
```

Run commands from the repository root. These scripts execute experiments and may take substantial time. Settings are primarily hard-coded in the scripts; `smfg_config.json` is a saved configuration, **not a universal command-line configuration loader**. Some original module docstrings refer to earlier names (`verify_convergence.py`, `regen_v3.py`, `smfg_experiments.py`); use the actual filenames above.

## Interpreting results

The code contains different exploitability estimators and iterate conventions. In particular, the finite-population Monte Carlo quantity in `code_revised.py` must not be silently equated with the proxy or homogeneous-deployment metric used by other scripts. Compare matching environments, seeds, estimators and iteration budgets. Saved outputs are historical artifacts and were not independently regenerated for this portfolio package.

## Validation and packaging

All included Python files passed syntax parsing on 22 September 2026. Numerical experiments were not rerun. Dependencies were inferred from imports and are not version-locked. The package excludes third-party papers, the internship brief, private tool metadata, LaTeX build products, archive duplicates and the 26 MB trajectory CSV. Synthetic environments require no bundled private dataset.

The original project did not include an explicit software license. This package preserves joint authorship and does not assert a new license. Author email addresses were removed from the portfolio copy of the report; the source report is unchanged.
