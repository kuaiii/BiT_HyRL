# Data dictionary

Each file is a tidy CSV: one row per evaluated cell, UTF-8, comma separated, `nan` where a
metric is undefined. Column names follow the manuscript (`r_res`, `auc_gcc`, `auc_csa`,
`auc_hc`), so a table can be rebuilt by grouping on `dataset` / `policy` (or `arm`) /
`attack` / `seed`.

Every builder in `scripts/` reads from this directory.

---

## `results/` (design point)

| File | Rows | Content |
|---|---|---|
| `input_sweep.csv` | 194 | Design-point sweep behind Fig. 2: for a grid of `n` and `⟨k⟩` the admissible window `[c_min, c_dir]`, the optimal `k_max` and the realised mean degree. Consumed by `gen_fig_input.py`. |
| `input_sweep_fallback.csv` | 5 | Measured `k_max = k⁻` fallback for the cells where the admissible window is empty (the dashed segment of Fig. 2). |

## `results/family/` — thirty-topology deployment experiment

| File | Rows | Content |
|---|---|---|
| `family_ctrl.csv` | 16 | Table 6 body. Mean `R_res` of every strategy overall and per family (BA, ER, WS, ISP, AS) plus the three component areas, on all thirty topologies. `BiT-HyRL` = 0.4040. |
| `family_paired.csv` | 15 | Table 7. Per-topology paired differences against BiT-HyRL, their 95 % Student-`t` intervals, and Holm-corrected two-sided Wilcoxon `p` values, for each of the four metrics. |
| `_family_ctrl_meanstd.csv` | 16 | Same rows with seed-to-seed standard deviations — the `mean ± std` printing of Table 6. |
| `family_bench.csv` | 1950 | Construction-only benchmark (no controller, no cascade): four curve areas, `κ`, construction wall clock, per `(graph, construction, build seed)`. Source of Table 8 and of Fig. 5–6. |
| `agg_method.csv` | 13 | Per-method means of `family_bench.csv` (`A`, the four areas, `κ`, `⟨k⟩`, construction time). |
| `agg_family.csv` | 13 | The same means split by family. |
| `fig_points.csv` | 390 | One row per `(graph, construction)`: `A` against `κ` and `⟨k⟩` — panel (a)(b) of Fig. 6. |
| `fig_family_A.csv` | 78 | Family means of `A` — the bar panel of Fig. 5. |
| `curves_BA-100.csv` | 9 090 | Stored `GCC` / `E_glob` curves of the eight constructions on BA-100 under both removal orders — panels (a)(b) of Fig. 5. |
| `construction_seed.csv` | 1 080 | Sensitivity of the deployed arm to the construction seed (reported alongside Table 13): `r_res` for 30 graphs × 2 arms × 3 build seeds × 2 attacks × 3 eval seeds. |
| `construction_seed_summary.csv` | 30 | Per-graph summary of the above: baseline, BiT mean/min/max, worst-case gain, seed spread. |
| `attack_orders/attack_orders_summary.csv` | 48 | Ranking of the eight constructions under six removal orders (Fig. 5, panel e): mean `A`, its CI, mean rank, `p(rank = 1)`. |
| `attack_orders/attack_orders_detail.csv` | 7 200 | The per-cell rows behind that summary (30 graphs × 8 methods × 5 seeds × 6 orders), with `n`, `m`, `κ`, `k_max`, construction time. |
| `attack_orders/attack_orders_paired.csv` | 12 | Paired `A` differences against UNITY for each order. |
| `attack_orders/attack_orders_runtime.csv` | 240 | Construction wall clock per `(method, order, seed)`. |
| `attack_orders/attack_orders_manifest.json` | — | Which shard produced which file (provenance). |

## `results/baselines/` — compared methods, run through the same harness

| File | Rows | Content |
|---|---|---|
| `curve_archive.csv` | 560 | Fig. 4 source. Every curve of the two panels: 2 datasets (BA-100, Colt) × 7 methods × 2 orders × 20 seeds, with `f_grid`, `r_curve`, `csa_curve`, `hc_curve` stored as JSON lists. |
| `method_matrix/matrix_scalar.csv` | 10 200 | Per-cell scalar records of the compared constructions and placements on the thirty graphs (six orders × 20 seeds). |
| `method_matrix/placement_new/matrix_scalar.csv` | 2 400 | The same for the re-run placement proxies (MWCL, MADQN, k-median, random). |
| `ablation/ablation_all_scalar.csv` | 3 600 | Table 9 source. Three substrate arms (original, degree-preserving re-randomisation, reconstructed) × 30 graphs × 2 attacks × 20 seeds, with the deployed policy. |
| `romen/romen_9x6.csv` | 1 080 | ROMEN re-run on the nine attack-coverage graphs, all six orders (ROMEN is a stochastic GA, so the coverage table uses this run rather than the archive). |
| `runtime_benchmark.csv` | 37 | Table 10 source. Deployment-stage wall clock: reconstruction + placement on one core, per BA scale. |
| `runtime_benchmark_srv.csv` | 40 | The same measured on the batch server (used for the cross-machine note). |
| `runtime_romen.csv`, `runtime_romen_small.csv` | 1, 4 | ROMEN construction time at the scales where its sweep is memory-bound. |

## `results/attack_cov/`

| File | Rows | Content |
|---|---|---|
| `attack_coverage.csv` | 11 880 | Table 11 source. Merged coverage grid: 9 topologies × 11 strategies × 6 removal orders (`degree`, `betweenness`, `closeness`, `coreness`, `eigenvector`, `random`) × 20 seeds, with `r_res` and the three areas. |

## `results/advanced/`

| File | Rows | Content |
|---|---|---|
| `advanced_attacks.csv` | 480 | Table 12 source. Four advanced attack models (adaptive, correlated, multi-stage, GDM) × 8 arms × 5 BA scales × 3 seeds. |

## `results/budget/`

| File | Rows | Content |
|---|---|---|
| `budget_all.csv` | 1 080 | Table 13 source. Controller deployment ratio `ρ_c ∈ {0.05, 0.10, 0.20}` × 30 graphs × 2 arms × 2 attacks × 3 eval seeds. |

## `results/capacity/`, `results/scale/`

| File | Rows | Content |
|---|---|---|
| `capacity/capacity.csv` | 360 | Capacity-model sweep (load margin `θ`) quoted in the text: per-cell `r_res`, collapse fraction, iterations to convergence. |
| `scale/c_window.csv` | 44 | The admissible `c` window per graph: `c`, `k_max`, area, and whether the point is the deployed one. |
| `scale/c_tradeoff.csv` | 117 | Efficiency-vs-resilience trade-off along `c` (30 graphs × 3–4 points): `auc_degree`, `auc_random`, `E_0`. |
| `scale/dense_upper_bound.csv` | 15 | Dense-substrate probe: `R_res` of the deployed arm at `⟨k⟩ ≈ 20`, with seed spread and standard error. |

## `results/checkpoints/`

| File | Content |
|---|---|
| `ppo_final_s0.pt` | The frozen placement policy used for every reported number: GAT encoder + PPO actor-critic, saved with its config, so `bit_hyrl.rl.transfer.load_policy` can rebuild it without the training corpus. |

---

## Not included

* `family/ctrl/`, `family/ctrl_romen/`, `attack_cov/` (per-cell curve shards, about 130 MB):
  the merged tables above reproduce every printed number; the shards are only needed to
  rebuild those merges. Available from the authors.
* `records/` (per-cell JSON, about 312 MB): superseded by the CSVs above.

## Regenerating the merged tables from scratch

`scripts/run_family.py` + `scripts/family_bench.py` + `scripts/family_ctrl_report.py`
rebuild `family_ctrl.csv` / `family_paired.csv` from a fresh campaign;
`scripts/attack_coverage.py` rebuilds `attack_coverage.csv` from the shard directories.
Both write into the paths above, and `scripts/check_package.py` re-verifies the headline
numbers afterwards.
