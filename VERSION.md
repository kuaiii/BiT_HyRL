# Version record for JRESS-D-26-02993

Release date: 2026-11-19
Repository: https://github.com/kuaiii/BiT_HyRL
Package version: bit_hyrl 1.0.0

## Pinned artifact checksums

| Artifact | Path | Size in bytes | SHA-256 |
|---|---|---:|---|
| Frozen BiT-HyRL PPO policy | `results/checkpoints/ppo_final_s0.pt` | 97,897 | `98e2d91235c46e1819188c91e310686bc53cd8c74102b8840dfe872370660fe3` |
| GDM released weights | `network_dismantling/GDM/models_newpg/gdm_nx_best.pth` | 34,321 | `d9e916d189eb27a24bbc30df610c0179e5f53f46afd4f357899a0363e16652c7` |

Verify with:

```bash
sha256sum results/checkpoints/ppo_final_s0.pt
sha256sum network_dismantling/GDM/models_newpg/gdm_nx_best.pth
```

## Frozen training run

The deployed checkpoint is the final iterate of a fixed training schedule. Its embedded
configuration records 200 PPO updates, six episodes per update, seed 0, early-stopping
patience 0 and 3,583.364 training seconds. Validation was monitored as a diagnostic and was
not used to select the deployed checkpoint.

The exact retraining command is:

```bash
python -m bit_hyrl train \
    --split configs/splits.json \
    --iterations 200 \
    --episodes 6 \
    --seed 0 \
    --patience 0 \
    --eval-sample 20 \
    --out results/checkpoints/ppo_final_s0.pt
```

## Archived results

The archived tables reproduce every reported number. Run the standard-library integrity
check with:

```bash
python scripts/check_package.py
```

A successful run ends with:

```text
all 12 checks passed — the package reproduces the reported numbers.
```
