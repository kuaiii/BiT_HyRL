# BiT-HyRL — code and data for the manuscript

*Bimodal Topology and Hybrid Reinforcement Learning for Resilient Distributed Control
Systems Against Cascading Failures* — Reliability Engineering & System Safety,
manuscript **JRESS-D-26-02993**.

This package contains everything behind the numbers in the manuscript: the implementation
of the method, the thirty evaluation topologies, the archived experiment results, the
seven figures and the LaTeX fragments of the fourteen numerical tables.

It has two intended uses:

1. **Read and verify** — every table and figure of the paper can be regenerated from the
   archived results in seconds, on a laptop, without re-running a single cascade
   simulation and without a GPU.
2. **Re-run** — the experiment drivers that produced the archive are included, so the
   whole campaign can be repeated from scratch (a multi-core CPU box, hours to days
   depending on the subset).

---

## 1. Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) integrity check: 11 assertions on the archived data, standard library only, ~1 s
python scripts/check_package.py

# 2) regenerate the numerical tables into paper_tables/
python scripts/family_ctrl_report.py                     # Table 6, Table 7
python scripts/family_report.py                          # Table 8
python scripts/make_paper_tables.py --out paper_tables      # Tables 3, 7, 9, 10, 12, 13, 14 (+2 fragments not used in the manuscript)
python scripts/table_attack_coverage.py                     # Table 11

# 3) regenerate the data figures into figures/
python scripts/paper_figures.py --out figures --only trends --with-domain   # Fig. 4, Fig. 7
python scripts/fig_construction.py --out figures                     # Fig. 5, Fig. 6
python scripts/gen_fig_input.py --out figures/fig_input_sensitivity.pdf \
       --png figures/fig_input_sensitivity.png                              # Fig. 2
```

`paper_tables/` already holds the fragments produced by exactly these commands, so the
regenerated files can be diffed against them. The submitted `main.tex` inlines the same
numbers and remains the authoritative version.

---

## 2. Layout

```text
BiT-HyRL_RESS_code_and_data/
├── bit_hyrl/              # the method: bimodal construction, RL placement, cascade, metrics
│   ├── topology/             #   BiT: admissible degree window, plan, Havel-Hakimi realisation
│   ├── rl/                   #   Hy+RL: GAT encoder, PPO training, policy transfer/inference
│   ├── control/              #   controller assignment, capacity model, cascade dynamics
│   ├── evaluation/           #   GCC / CSA / H_C / R_res, curve areas, evaluation runner
│   ├── attack/               #   six removal orders + four advanced attacks (adaptive, correlated, multi-stage, GDM)
│   ├── analysis/             #   aggregation, statistics, ablation and matrix drivers
│   └── io/, config/, rng.py  #   graph loading + fingerprinting, config schema, deterministic RNG
├── network_dismantling/      # vendored third-party dismantling library (GDM weights included), used by the advanced attacks
├── configs/                  # experiment configuration
│   ├── graphs.json           #   the thirty evaluation graphs with (n, m) and edge-list SHA-256
│   ├── splits.json           #   train/val/test split of the training corpus
│   ├── method.yaml           #   PPO and network hyperparameters (manuscript Table 5)
│   └── attacks.yaml, datasets.yaml
├── dataset/testdata/         # the thirty topologies (GML), as loaded by graphs.json
├── scripts/                  # table/figure builders, experiment drivers, integrity check
├── results/                  # archived results — the single data root, see results/README.md
│                             #   family/, baselines/, advanced/, budget/, capacity/, scale/,
│                             #   attack_cov/, checkpoints/
├── figures/                  # the seven figures of the manuscript (PDF, plus PNG previews)
└── paper_tables/             # LaTeX fragments of the fourteen numerical tables
```

**Naming.** `bit_hyrl` is the import root of the method package; `results/` is the single
data root that every script reads from and writes to, and `configs/` holds the pinned
experiment configuration. No other data version is referenced anywhere in the code.

---

## 3. The method in one page

**Substrate (BiT).** For a target node count `n`, edge count `m` and mixing parameter `c`,
the construction splits the nodes into a low-degree and a high-degree group and realises
the resulting degree sequence by Havel-Hakimi with a downgrade loop, so every candidate is
graphical by construction. A monotone sweep of `c` yields an admissible interval
`[c_min, c_dir]`: below `c_min` the cascade survives on structure alone, above `c_dir` the
directed-efficiency ceiling is violated. The deployed design point is the geometric
midpoint

```
c = sqrt(c_min · c_dir)
```

with `k_max = round(sqrt(k_max^- · k_max^+))` on a non-empty window, and the *lower* bound
with a reported `λ*` when the window is empty. The two endpoints are kept in the code as
the ablation variants **BiT-R** (`c_min`, robustness-first) and **BiT-E** (`c_dir`,
efficiency-first); the paper deploys the geometric midpoint only. The historical label
`BiT-A` refers to the same construction and is still accepted when reading old records.

**Placement (Hy + RL).** Controllers are placed by a graph-attention encoder followed by a
PPO policy with a feasibility mask, trained on the synthetic corpus of
`configs/splits.json` and frozen for all reported numbers
(`results/checkpoints/ppo_final_s0.pt`). Placement never observes how the substrate was
built, which is what makes the topology and placement ablations in Table 9 meaningful.

**Evaluation.** A controller-aware cascade runs on the reconstructed graph under six
removal orders; `R_res` is the weighted mean of the normalised areas under the `GCC`, `CSA`
and `H_C` curves. Every number in the paper is a mean over the twenty campaign seeds of
each `(topology, attack)` cell.

---

## 4. Where each number comes from

### Tables (numbering as in the submitted `main.tex`)

| # | LaTeX label | Content | Produced by | Read from |
|---|---|---|---|---|
| 1 | `tab:notation` | notation | — (manual) | — |
| 2 | `tab:metrics` | metric definitions | — (manual) | — |
| 3 | `tab:datasets` | evaluation suite | `make_paper_tables.py` | `configs/graphs.json` |
| 4 | `tab:baselines` | compared methods | — (manual) | — |
| 5 | `tab:ppo_hyperparameters` | PPO / network settings | — (manual, mirrors `configs/method.yaml`) | `configs/method.yaml` |
| 6 | `tab:family_comparison` | `R_res` by family, thirty topologies | `family_ctrl_report.py` | `results/family/family_ctrl.csv` |
| 7 | `tab:statistical_tests` | paired tests against BiT-HyRL | `family_ctrl_report.py`, `make_paper_tables.py` | `results/family/family_paired.csv` |
| 8 | `tab:construction_budget` | construction-only comparison at fixed `(n, m)` | `family_report.py` | `results/family/family_bench.csv` |
| 9 | `tab:ablation` | original / re-randomised / reconstructed substrate | `make_paper_tables.py` | `results/baselines/ablation/ablation_all_scalar.csv` |
| 10 | `tab:runtime` | deployment-stage wall clock | `make_paper_tables.py` | `results/baselines/runtime_benchmark.csv` |
| 11 | `tab:attack_coverage` | six removal orders, nine topologies | `table_attack_coverage.py` | `results/attack_cov/attack_coverage.csv` |
| 12 | `tab:advanced_attacks` | four advanced attack models | `make_paper_tables.py` | `results/advanced/advanced_attacks.csv` |
| 13 | `tab:sensitivity_ratio` | controller deployment ratio `ρ_c` | `make_paper_tables.py` | `results/budget/budget_all.csv` |
| 14 | `tab:sensitivity_weights` | weighting of the `R_res` aggregate | `make_paper_tables.py` | `results/family/family_ctrl.csv` (reweighted in place) |

### Figures

| # | File in `figures/` | Content | Produced by |
|---|---|---|---|
| 1 | `Theory Foundation.pdf` | substrate, failure modes, cascade | conceptual schematic — no code |
| 2 | `fig_input_sensitivity.pdf` | design point vs. `n` and `⟨k⟩` | `gen_fig_input.py` ← `results/input_sweep.csv` (+ `input_sweep_fallback.csv`) |
| 3 | `Algorithms.pdf` | architecture and algorithm flow | conceptual schematic — no code |
| 4 | `Bimodal network enhance resilience.pdf` | `GCC`, `CSA`, `H_C` trends on BA-100 and Colt | `paper_figures.py --only trends` ← `results/baselines/curve_archive.csv` |
| 5 | `construction_overview.pdf` | eight constructions at fixed budget | `fig_construction.py` ← `family/curves_BA-100.csv`, `family_bench.csv`, `agg_method.csv`, `fig_points.csv`, `attack_orders/attack_orders_summary.csv` |
| 6 | `construction_mechanism.pdf` | why the wiring helps, and what it costs | `fig_construction.py` (same inputs) |
| 7 | `applicability_domain.pdf` | margin vs. mean degree, thirty topologies | `paper_figures.py --with-domain` ← `family/family_ctrl.csv`, `baselines/` |

`paper_figures.py --only comparison` additionally draws `Comparison_topology.pdf`, an
earlier rendering of the construction comparison (grouped bars in panel d). It is kept in
the script for reference but is **not** the version used in the manuscript; Fig. 5 is the
`construction_overview` one.

---

## 5. Data

`results/README.md` documents every file: what it is, how many rows, which figure or table
consumes it. The short version:

| Path | What it is |
|---|---|
| `results/family/` | the thirty-topology deployment experiment: main table, paired tests, construction benchmark, attack-order ranking, and the curves of Fig. 5–6 |
| `results/baselines/` | per-cell records of the compared methods, the ROMEN re-run, the runtime benchmark and the curve archive of Fig. 4 |
| `results/advanced/` | the four advanced attack models (Table 12) |
| `results/attack_cov/` | the merged attack-coverage grid, 11 methods × 6 orders × 9 graphs × 20 seeds (Table 11) |
| `results/budget/` | controller-budget sweep (Table 13) |
| `results/capacity/`, `results/scale/` | capacity-model sweep and design-window data quoted in the text |
| `results/checkpoints/` | the frozen PPO policy `ppo_final_s0.pt` |
| `results/input_sweep*.csv` | the design-point sweep behind Fig. 2 |

All tables are long/tidy CSVs: one row per `(dataset, policy, attack, seed)` cell, with
both the scalar areas and, where relevant, the stored curves.

---

## 6. Re-running the experiments

The drivers below repeat a campaign from scratch. Two of them (`attack_coverage.py` and the
shard-level family drivers) read the per-cell shard directories that are **not** shipped
(§7); every table and figure builder in §1 works from the merged tables alone and never
needs them.

| Campaign | Driver | Manuscript object |
|---|---|---|
| training-corpus split | `build_splits.py`, `generate_training_data.py` | §5.1 training protocol, Algorithm 1 |
| deployment: 30 graphs × 3 variants × 2 attacks × 20 seeds | `run_family.py` (shard scheduler) + `family_bench.py`, then `family_ctrl_report.py` | Tables 6–7 |
| construction benchmark, no controller | `family_bench.py` (methods without `BiT-*`) + `run_family.py`, then `family_report.py` | Table 8 |
| ablation arms | `eval_ablation_arms.py` | Table 9 |
| attack coverage (9 graphs × 6 orders) | `attack_coverage.py` | Table 11 |
| advanced attacks | `advanced_attacks.py` | Table 12 |
| controller budget / construction seeds | `controller_budget_sensitivity.py`, `construction_seed_sensitivity.py` | Tables 13 and 8 |
| dense-substrate probe | `dense_upper_bound.py` | §5.6 |
| design window, admissible interval | `bimodal_solve.py`, `param_window.py`, `fc_robust.py`, `isp_construct.py`, `input_sweep.py`, `input_sweep_fallback.py` | §3.2–3.4, Fig. 2 |
| curve archive for Fig. 4 | `export_curves.py` | Fig. 4 |

`bit_hyrl` also exposes a command line of its own
(`python -m bit_hyrl --help`): `fingerprint`, `construct`, `sweep-c`, `evaluate`,
`train`, `verify`.

---

## 7. What is *not* in this package, and why

| Omitted | Size | Why | How to obtain |
|---|---|---|---|
| per-cell JSON records (`results/records/`, 13 710 files) | ≈ 312 MB | superseded by the tidy CSVs shipped here; kept on the experiment server | from the authors on request |
| per-cell shards behind Tables 6 and 11 (`family/ctrl*`, `attack_cov*`) | ≈ 130 MB | the merged tables shipped here reproduce every reported number; the shards are only needed to rebuild those merges | from the authors on request |
| training corpus (`dataset/all`, ≈ 12 k synthetic graphs) | ≈ GB | generated on demand | `scripts/generate_training_data.py` |
| standalone upstream repositories of the compared constructions (UNITY, ROMEN, ONION, QDLM, FRED-ABL, Eff-R-Net) | — | the comparison is run through the single-protocol ports in `bit_hyrl/baselines/`, so that every method sees the same cascade, metrics and seeds; the upstream trees are not redistributed | <https://github.com/kuaiii/dismantling_topology> and <https://github.com/kuaiii/redesign_topology> |

Only the thin ports in `bit_hyrl/baselines/` are shipped, and they exist solely so that
Table 6 and Fig. 5 compare like with like. `network_construction/`, `network_metrics/` and
the legacy `src/` pipeline of earlier revisions are deliberately absent: nothing in this
package imports them.

---

## 8. Environment

* Python ≥ 3.9, CPU only for every table/figure command above.
* `torch` is required only to load the checkpoint or to train a new policy; the archived
  numbers do not need it.
* `torch_sparse` / `torch_scatter` are required only to re-run the GDM advanced attack;
  `python scripts/check_package.py` and all table builders work without them.
* Determinism: every campaign derives its seeds from `(dataset, attack, seed)` through
  `bit_hyrl/rng.py`; no global seed is set anywhere, so shards can be run in any order
  and on any number of workers.

---

## 9. Third-party material

* `network_dismantling/` is a vendored copy of the dismantling library released with
  *Robustness and resilience of complex networks* (Artime et al.), GPL-3.0, used unchanged
  for the four advanced attack models; the GDM weights file is included.
* The thirty evaluation graphs are the synthetic BA/ER/WS substrates built for this study
  plus real topologies from the SNAP `as-733` BGP-AS snapshots and the Topology Zoo ISP
  backbones, normalised to simple undirected graphs; `configs/graphs.json` pins `(n, m)`
  and the SHA-256 of the sorted edge list of each of them.

---

## 10. Availability and contact

This package is published at <https://github.com/kuaiii/BiT_HyRL>. The compared constructions
are released as separate repositories: <https://github.com/kuaiii/dismantling_topology> and
<https://github.com/kuaiii/redesign_topology>.

Correspondence about this package: the corresponding author of JRESS-D-26-02993.

---

## 11. License

Copyright (C) 2026 Yizhong Hu, Jianfeng Guan, Su Yao, Junxiang Jia, Mu Wang, Changqiao Xu.

This package is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version. It is distributed in the hope that it
will be useful, but **without any warranty**; without even the implied warranty of
merchantability or fitness for a particular purpose. The full text is in [`LICENSE`](LICENSE).

`network_dismantling/` is a vendored third-party library (see §9) that is itself released
under GPL-3.0, so it is redistributed here under the same terms as the rest of the package.
