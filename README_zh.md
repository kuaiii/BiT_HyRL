# BiT-HyRL 代码与数据包（中文说明）

对应稿件：*Bimodal Topology and Hybrid Reinforcement Learning for Resilient Distributed
Control Systems Against Cascading Failures*，RESS，稿号 **JRESS-D-26-02993**。

本包只装**本文方法**及其自身实验：BiT-HyRL 的实现、30 张评测拓扑、论文全部数值的结果档、
7 张插图、14 张数值表的 LaTeX 片段。对比方法的独立实现不在本包内（另有库），
只在 `bit_hyrl/baselines/` 保留了"同一套级联/指标/种子口径"的薄封装，用来保证第 6 张表
和 Fig. 5 是同一口径下的比较。

评审/读者拿到包可以做两件事：

1. **不重跑实验**，直接从结果档重算出论文每张表、每张图（笔记本 CPU、几秒钟）。
2. **重跑实验**，用 `scripts/` 里的驱动脚本复现整个实验战役（多核 CPU，按子集几小时到几天）。

---

## 一、三步验证

```bash
pip install -r requirements.txt

# 1) 完整性自检：11 项断言，只用标准库，约 1 秒
python scripts/check_package.py

# 2) 重出数值表（写入 paper_tables/）
python scripts/family_ctrl_report.py                  # 表 6、表 7
python scripts/family_report.py                       # 表 8
python scripts/make_paper_tables.py --out paper_tables   # 表 3、7、9、10、12、13、14
python scripts/table_attack_coverage.py                  # 表 11

# 3) 重出插图（写入 figures/）
python scripts/paper_figures.py --out figures --only trends --with-domain   # Fig. 4、Fig. 7
python scripts/fig_construction.py --out figures                     # Fig. 5、Fig. 6
python scripts/gen_fig_input.py --out figures/fig_input_sensitivity.pdf \
       --png figures/fig_input_sensitivity.png                              # Fig. 2
```

`paper_tables/` 里放的就是上面这些命令的产物，可以直接 diff。**以投稿的 `main.tex` 为准**，
它把同样的数字内联在正文里。

自检脚本会核对：30 张图与清单一致、`R_res = 0.4040`、对 UNITY 的配对差 `+0.062`（18.1%，
`p = 1.6e-5`）、外部方法全部 `p < 0.001`、各战役的网格规模（560 / 11 880 / 480 / 1 080 行）、
7 张图与 14 个表片段是否齐全。

---

## 二、目录

```text
BiT-HyRL_RESS_code_and_data/
├── bit_hyrl/          方法本体：双峰构造（topology/）、RL 布置（rl/）、级联（control/）、
│                         指标与评测（evaluation/）、攻击与高级攻击（attack/）、聚合统计（analysis/）
├── network_dismantling/  第三方拆解库（含 GDM 权重），只为四种高级攻击而内置
├── configs/              实验配置：graphs.json（30 图清单 + n/m + 边表 SHA-256）、
│                         splits.json（训练/验证/测试划分）、method.yaml（PPO 超参，对应表 5）
├── dataset/testdata/     30 张评测拓扑（GML）
├── scripts/              出表、出图、实验驱动、完整性自检
├── results/              结果档（唯一数据根），逐文件说明见 results/README.md
├── figures/              论文 7 张插图（PDF，另附 PNG 预览）
└── paper_tables/         14 张数值表的 LaTeX 片段
```

**关于目录名**：`bit_hyrl` 是方法包的导入根目录（`import bit_hyrl`），`results/` 是唯一数据根，
所有脚本都从这里读、往这里写；`configs/` 是固定下来的实验配置。代码里不再出现任何内部版本号。

---

## 三、方法要点（一句话版）

- **构造（BiT）**：在度序列层面把节点分成低度/高度两组，用 Havel–Hakimi 加降级回路实现，
  保证每个候选都可图；对混合参数 `c` 扫描得到可容许区间 `[c_min, c_dir]`，部署点取**几何中点**
  `c = sqrt(c_min · c_dir)`。两端点只作消融变体：**BiT-R**（`c_min`，鲁棒优先）、
  **BiT-E**（`c_dir`，效率优先）；历史标签 `BiT-A` 与 `BiT-HyRL` 指同一方法。
- **布置（Hy+RL）**：GAT 编码 + 带可行性掩码的 PPO，把控制器放到图上；策略对"图是怎么造的"
  不可见，所以表 9 的基板消融才有意义。
- **评测**：控制器感知的级联 × 六种移除顺序，`R_res` 是 `GCC`/`CSA`/`H_C` 曲线面积加权；
  每个 `(图, 攻击)` 单元 20 个种子。

---

## 四、表与图的数据来源（速查）

| 论文对象 | 脚本 | 数据 |
|---|---|---|
| 表 6、表 7（主表 + 配对检验） | `family_ctrl_report.py` | `family/family_ctrl.csv`、`family_paired.csv` |
| 表 8（同预算构造对比） | `family_report.py` | `family/family_bench.csv` |
| 表 9（基板消融） | `make_paper_tables.py` | `baselines/ablation/ablation_all_scalar.csv` |
| 表 10（部署耗时） | `make_paper_tables.py` | `baselines/runtime_benchmark.csv` |
| 表 11（攻击覆盖） | `table_attack_coverage.py` | `attack_cov/attack_coverage.csv` |
| 表 12（四种高级攻击） | `make_paper_tables.py` | `advanced/advanced_attacks.csv` |
| 表 13（控制器预算） | `make_paper_tables.py` | `budget/budget_all.csv` |
| 表 14（权重敏感性） | `make_paper_tables.py` | `family/family_ctrl.csv` |
| Fig. 2（设计点敏感性） | `gen_fig_input.py` | `results/input_sweep*.csv` |
| Fig. 4（GCC/CSA/H_C 轨迹） | `paper_figures.py` | `baselines/curve_archive.csv` |
| Fig. 5、Fig. 6（构造对比与机理） | `fig_construction.py` | `family/curves_BA-100.csv`、`family_bench.csv`、`fig_points.csv`、`attack_orders/` |
| Fig. 7（适用性边界） | `paper_figures.py --with-domain` | `family/family_ctrl.csv` + `baselines/` |

表 1、表 2（记号、指标定义）、表 4（对比方法清单）、表 5（超参）为人工维护；Fig. 1、Fig. 3
为示意图，无对应代码。

---

## 五、有意没有放进去的内容

| 未包含 | 体积 | 原因 |
|---|---|---|
| 逐格曲线分片（`family/ctrl*`、`attack_cov*`） | ≈ 130 MB | 上文这些合并表已能复现全部纸面数值；分片只用于重建合并表，可向作者索取 |
| 逐格 JSON 记录（`records/`，13 710 个文件） | ≈ 312 MB | 旧版本数据，已被 CSV 取代，留在实验服务器 |
| 训练语料 `dataset/all`（约 1.2 万张合成图） | GB 级 | 按需生成：`scripts/generate_training_data.py` |
| 对比方法的上游独立仓库 | — | 比较统一走 `bit_hyrl/baselines/` 的薄封装，上游代码不再分发：<https://github.com/kuaiii/dismantling_topology>、<https://github.com/kuaiii/redesign_topology> |

旧版管线（`src/`）、旧指标树（`network_metrics/`）、旧基线树（`network_construction/`）已全部剔除，
本包内没有任何文件 import 它们。

---

## 六、环境

- Python ≥ 3.9；上面的出表/出图命令全部 CPU 即可。
- `torch` 只在加载 checkpoint 或重训策略时需要。
- `torch_sparse` / `torch_scatter` 只在重跑 GDM 攻击时需要；自检与出表都不依赖。
- 随机性：所有种子由 `(dataset, attack, seed)` 经 `bit_hyrl/rng.py` 派生，代码里不设全局种子，
  因此分片可以任意顺序、任意并行度跑，结果逐位一致。

---

## 七、代码地址

- 本包：<https://github.com/kuaiii/BiT_HyRL>
- 对比方法：<https://github.com/kuaiii/dismantling_topology>、<https://github.com/kuaiii/redesign_topology>

---

## 八、许可

版权所有 (C) 2026 Yizhong Hu、Jianfeng Guan、Su Yao、Junxiang Jia、Mu Wang、Changqiao Xu。

本包为自由软件：你可以依据自由软件基金会发布的 GNU 通用公共许可证（GPL）第 3 版或（由你选择）
任何更新版本的条款，再分发和/或修改本包。分发本包是希望它有用，但**不提供任何担保**，甚至不提供
可商用性或特定用途适用性的默示担保。完整条款见 [`LICENSE`](LICENSE)，版权声明与第三方素材
归属见 [`NOTICE`](NOTICE)。

`network_dismantling/` 是第三方库（见第五节），其自身即按 GPL-3.0 发布，故在本包中按同一许可证再分发。
