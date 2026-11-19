# BiT-HyRL

Code and data for the manuscript *Bimodal Topology and Hybrid Reinforcement Learning for
Resilient Distributed Control Systems Against Cascading Failures* (Reliability Engineering
& System Safety, JRESS-D-26-02993).

The repository contains the implementation of the method, the thirty evaluation topologies,
the archived results behind every table and figure, the seven figures and the LaTeX
fragments of the fourteen numerical tables.

Requirements: Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`torch` is needed only to train a policy or to load the checkpoint; `torch_sparse` and
`torch_scatter` only to re-run the GDM attack.

## Verifying the archive

```bash
python scripts/check_package.py
```

This recomputes the headline numbers from the archived CSV files using the standard
library only: the thirty graphs against their pinned fingerprints, `R_res = 0.4040`, the
paired difference against UNITY, the row counts of the campaign grids, the presence of
the seven figures and fourteen tables, and that the training defaults in `bit_hyrl/cli.py`
and `configs/default.json` still match the run that produced the shipped checkpoint.
Exit code 0 means every check passed.

## Layout

```text
bit_hyrl/                  method package
  topology/                  bimodal construction: degree plan, Havel-Hakimi, decorrelation
  rl/                        graph-attention encoder, PPO training, policy transfer
  control/                   controller assignment, capacity model, cascade
  evaluation/                GCC, CSA, H_C, R_res; evaluation runner
  attack/                    six removal orders and four advanced attacks
  analysis/                  aggregation, paired statistics, ablation and matrix drivers
  io/, config/, rng.py       graph loading and fingerprints, config schema, seed derivation
network_dismantling/       third-party dismantling library, used by the GDM attack
configs/                   graphs.json, splits.json, method.yaml, attacks.yaml, datasets.yaml,
                           default.json (reference copy of the training run, not an input)
dataset/testdata/          the thirty evaluation topologies (GML)
scripts/                   experiment drivers, table and figure builders, integrity check
results/                   archived results; see results/README.md
figures/                   the seven manuscript figures
paper_tables/              LaTeX fragments of the fourteen numerical tables
```

## Method

*Substrate.* For a target node count `n`, edge count `m` and mixing parameter `c`, the
construction splits the nodes into a low-degree and a high-degree group and realises the
resulting degree sequence with Havel-Hakimi plus a downgrade loop, so every candidate is
graphical by construction. Sweeping `c` gives an admissible interval `[c_min, c_dir]`:
below `c_min` the cascade survives on structure alone, above `c_dir` the directed
efficiency ceiling is violated. The deployed point is the geometric midpoint
`c = sqrt(c_min * c_dir)`, with the lower bound and a reported `lambda*` when the interval
is empty. The two endpoints are available as the variants `BiT-R` (`c_min`) and `BiT-E`
(`c_dir`).

*Placement.* Controllers are placed by a graph-attention encoder followed by a PPO policy
with a feasibility mask, trained on the corpus described by `configs/splits.json` and
frozen for all reported numbers (`results/checkpoints/ppo_final_s0.pt`). The placement
policy does not observe how the substrate was built.

*Evaluation.* A controller-aware cascade runs on the reconstructed graph under six removal
orders. `R_res` is the weighted mean of the normalised areas under the `GCC`, `CSA` and
`H_C` curves, averaged over the twenty seeds of each `(topology, attack)` cell.

## Training the policy

The shipped checkpoint is the one and only policy behind every reported number. It was
produced by this command, and by no other:

```bash
python -m bit_hyrl train \
    --split configs/splits.json \
    --iterations 200 --episodes 6 --seed 0 --patience 0 \
    --out results/checkpoints/ppo_final_s0.pt
```

| Setting | Value | Where it is fixed |
|---|---|---|
| PPO updates | 200 | `--iterations` |
| Rollout episodes per update | 6 | `--episodes` (default 6) |
| Total episodes / substrates | 1,200 / 1,200 | 200 x 6, drawn without replacement from the 2,000-graph training split |
| Early stopping | disabled, patience 0 | `--patience` (default 0) |
| Learning rate | 1e-4 | `--lr` (default 1e-4) |
| Discount factor | 0.99 | `--gamma` (default 0.99) |
| Reward weights `(w_G, w_S, w_H)` | 0.6 / 0.2 / 0.2 | `bit_hyrl.cli._PAPER_REWARD_WEIGHTS` |
| Clip ratio, GAE, entropy, epochs | 0.2, 0.95, 0.01, 4 | `TrainConfig` |
| Minibatch, value coefficient, grad clip | 32, 0.5, 0.5 | `cmd_train` |
| Reward attack window | first 30 % of the removal sequence | `TrainConfig.attack_prefix_fraction` |

The run took 3,583 s and ended at update 200 because that was the budget, so the saved
state is the final iterate; validation was logged but never used to select a checkpoint.

`configs/default.json` is a **reference copy, not an input**: `train` never reads it, since
`cmd_train` builds `TrainConfig` from the flags above. Its `train` block mirrors this table,
and `scripts/check_package.py` fails if the file and the CLI defaults drift apart again.
The authoritative machine-readable record is the `cfg` block inside the checkpoint itself:

```python
import torch
ckpt = torch.load("results/checkpoints/ppo_final_s0.pt", map_location="cpu", weights_only=False)
ckpt["cfg"], ckpt["patience"], ckpt["train_seconds"]
```

## Tables

| # | Label | Produced by | Read from |
|---|---|---|---|
| 3 | `tab:datasets` | `make_paper_tables.py` | `configs/graphs.json` |
| 6 | `tab:family_comparison` | `family_ctrl_report.py` | `results/family/family_ctrl.csv` |
| 7 | `tab:statistical_tests` | `family_ctrl_report.py`, `make_paper_tables.py` | `results/family/family_paired.csv` |
| 8 | `tab:construction_budget` | `family_report.py` | `results/family/family_bench.csv` |
| 9 | `tab:ablation` | `make_paper_tables.py` | `results/baselines/ablation/ablation_all_scalar.csv` |
| 10 | `tab:runtime` | `make_paper_tables.py` | `results/baselines/runtime_benchmark.csv` |
| 11 | `tab:attack_coverage` | `table_attack_coverage.py` | `results/attack_cov/attack_coverage.csv` |
| 12 | `tab:advanced_attacks` | `make_paper_tables.py` | `results/advanced/advanced_attacks.csv` |
| 13 | `tab:sensitivity_ratio` | `make_paper_tables.py` | `results/budget/budget_all.csv` |
| 14 | `tab:sensitivity_weights` | `make_paper_tables.py` | `results/family/family_ctrl.csv` |

Tables 1, 2, 4 and 5 are notation, metric definitions, the list of compared methods and the
hyperparameter table; they are set by hand, and Table 5 mirrors `configs/method.yaml`.

```bash
python scripts/family_ctrl_report.py
python scripts/family_report.py
python scripts/make_paper_tables.py --out paper_tables
python scripts/table_attack_coverage.py
```

## Figures

| # | File in `figures/` | Produced by |
|---|---|---|
| 1 | `Theory Foundation.pdf` | schematic, no code |
| 2 | `fig_input_sensitivity.pdf` | `gen_fig_input.py` from `results/input_sweep*.csv` |
| 3 | `Algorithms.pdf` | schematic, no code |
| 4 | `Bimodal network enhance resilience.pdf` | `paper_figures.py --only trends` from `results/baselines/curve_archive.csv` |
| 5 | `construction_overview.pdf` | `fig_construction.py` from `results/family/` |
| 6 | `construction_mechanism.pdf` | `fig_construction.py` from `results/family/` |
| 7 | `applicability_domain.pdf` | `paper_figures.py --with-domain` from `results/family/family_ctrl.csv` |

```bash
python scripts/paper_figures.py --out figures --only trends --with-domain
python scripts/fig_construction.py --out figures
python scripts/gen_fig_input.py --out figures/fig_input_sensitivity.pdf \
       --png figures/fig_input_sensitivity.png
```

`paper_figures.py --only comparison` still draws an earlier version of Fig. 5 (grouped bars
in panel d); it is kept for reference and is not used in the manuscript.

## Results

`results/README.md` documents each file, its row count and the table or figure that
consumes it.

* `results/family/` — the thirty-topology deployment experiment: main table, paired tests,
  construction benchmark, attack-order ranking, the curves of Figs. 5 and 6.
* `results/baselines/` — records of the compared methods, the ROMEN re-run, runtime
  benchmark, curve archive of Fig. 4.
* `results/advanced/` — the four advanced attacks (Table 12).
* `results/attack_cov/` — merged attack-coverage grid (Table 11).
* `results/budget/`, `results/capacity/`, `results/scale/` — sensitivity sweeps.
* `results/checkpoints/` — the frozen PPO policy.
* `results/input_sweep*.csv` — design-point sweep behind Fig. 2.

All tables are tidy CSVs, one row per `(dataset, policy, attack, seed)` cell.

## Re-running the experiments

| Campaign | Driver |
|---|---|
| training corpus and splits | `generate_training_data.py`, `build_splits.py` |
| deployment, 30 graphs x 3 variants x 2 attacks x 20 seeds | `run_family.py`, `family_bench.py`, then `family_ctrl_report.py` |
| construction benchmark, no controller | `family_bench.py`, `run_family.py`, then `family_report.py` |
| ablation arms | `eval_ablation_arms.py` |
| attack coverage | `attack_coverage.py` |
| advanced attacks | `advanced_attacks.py` |
| controller budget, construction seeds | `controller_budget_sensitivity.py`, `construction_seed_sensitivity.py` |
| dense substrate probe | `dense_upper_bound.py` |
| design window, admissible interval | `bimodal_solve.py`, `param_window.py`, `fc_robust.py`, `isp_construct.py`, `input_sweep.py`, `input_sweep_fallback.py` |
| curve archive for Fig. 4 | `export_curves.py` |

The method package also has a command line:

```bash
python -m bit_hyrl --help
```

with the sub-commands `fingerprint`, `construct`, `probe`, `sweep-c`, `feasibility`,
`verify`, `eval`, `train`, `transfer`, `ablate`, `matrix`, `method-matrix`,
`method-stats`, `stats` and `figures`.

Seeds are derived from `(dataset, attack, seed)` in `bit_hyrl/rng.py`; no global seed is
set, so shards can run in any order and on any number of workers.

## Not included

| Omitted | Why | How to obtain |
|---|---|---|
| per-cell JSON records (`results/records/`, 13 710 files) | superseded by the tidy CSVs | from the authors |
| per-cell shards behind Tables 6 and 11 (`family/ctrl*`, `attack_cov*`) | the merged tables reproduce every reported number | from the authors |
| training corpus (`dataset/all`, 23 757 graphs) | generated on demand, and too large for a git repository | <https://huggingface.co/datasets/huzhzh/Network_datasets>, or `scripts/generate_training_data.py` |
| upstream repositories of the compared constructions | the comparison runs through the ports in `bit_hyrl/baselines/`, so every method sees the same cascade, metrics and seeds | <https://github.com/kuaiii/dismantling_topology>, <https://github.com/kuaiii/redesign_topology> |

The ports in `bit_hyrl/baselines/` exist so that Table 6 and Fig. 5 compare like with like.

## Availability

This repository: <https://github.com/kuaiii/BiT_HyRL>.
Corpus and the GDM weights: <https://huggingface.co/datasets/huzhzh/Network_datasets>.
Compared constructions: <https://github.com/kuaiii/dismantling_topology> and
<https://github.com/kuaiii/redesign_topology>.

## License

Copyright (C) 2026 Yizhong Hu, Jianfeng Guan, Su Yao, Junxiang Jia, Mu Wang, Changqiao Xu.

Released under the GNU General Public License, version 3 or later; the full text is in
[`LICENSE`](LICENSE), the copyright statement and the third-party attribution in
[`NOTICE`](NOTICE). `network_dismantling/` is a third-party GPL-3.0 library and is
redistributed under the same terms.

The evaluation graphs are synthetic BA/ER/WS substrates built for this study plus real
topologies from the SNAP `as-733` snapshots and the Topology Zoo ISP backbones, normalised
to simple undirected graphs. `configs/graphs.json` pins `(n, m)` and the SHA-256 of the
sorted edge list of each graph.
