"""PPO trainer for controller placement."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np
import torch

from .. import rng as _rng
from ..config.schema import TrainConfig
from ..control.capacity import capacity_model
from ..io.graph_io import load_graph_spec
from .features import feature_config, feature_dim, node_features
from .gat import edge_index_from_graph
from .placement import PPOPolicy
from .ppo import ActorCritic, compute_returns, masked_distribution, surrogate_loss
from .reward import DEFAULT_REWARD_WEIGHTS, placement_reward

__all__ = ["train", "resolve_graphs", "build_model", "make_policy"]

LOGGER = logging.getLogger("bit_hyrl.rl.trainer")

#: ``(label, graph)`` pairs are resolved once; a spec is a graphs.json-style mapping.
GraphLike = Any


def _train_order(seed: int, n_graphs: int) -> list[int]:
    """Seed-derived permutation of the training graphs (sampling order for episodes)."""
    if n_graphs <= 1:
        return list(range(n_graphs))
    generator = _rng.derive(seed, "ppo", "train_order")
    return [int(index) for index in generator.permutation(n_graphs)]


def _init_with_generator(model: torch.nn.Module, generator: torch.Generator) -> None:
    """Re-initialise every RNG-dependent parameter from a **local** generator."""
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            bound = 1.0 / math.sqrt(max(1, module.in_features))
            torch.nn.init.uniform_(module.weight, -bound, bound, generator=generator)
            if module.bias is not None:
                torch.nn.init.uniform_(module.bias, -bound, bound, generator=generator)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)


def resolve_graphs(items: Sequence[GraphLike]) -> list[nx.Graph]:
    """Coerce a mix of specs / paths / ``nx.Graph`` objects into ``nx.Graph`` list."""
    graphs: list[nx.Graph] = []
    for item in items:
        if isinstance(item, nx.Graph):
            graphs.append(item)
        elif isinstance(item, dict) or hasattr(item, "__getitem__") and "path" in item:
            graphs.append(load_graph_spec(item)[0])  # type: ignore[arg-type]
        else:
            spec = {"path": Path(str(item))}
            graphs.append(load_graph_spec(spec)[0])
    return graphs


def build_model(
    feature_dim_value: int,
    hidden_dim: int,
    gcn_layers: int,
    heads: int = 2,
    seed: int = 0,
    device: str = "cpu",
) -> ActorCritic:
    """Build a :class:`ActorCritic` deterministically seeded from ``seed``."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        int(_rng.derive_int(seed, "ppo", "torch", low=0, high=2**31 - 1))
    )
    model = ActorCritic(feature_dim_value, hidden_dim, gcn_layers, heads=heads).to(device)
    _init_with_generator(model, generator)
    return model


def make_policy(model: ActorCritic, decode: dict[str, Any], device: str = "cpu") -> PPOPolicy:
    """Wrap ``model`` into a :class:`PPOPolicy` (used by train/transfer/eval)."""
    return PPOPolicy(model, decode, device=device)


def _rollout(
    model: ActorCritic,
    graph: nx.Graph,
    cfg: TrainConfig,
    episode_seed: int,
    *,
    include_betweenness: bool,
    betweenness_k: int,
    device: str,
) -> dict[str, Any]:
    """Collect one placement episode and its terminal reward."""
    n = graph.number_of_nodes()
    model_cap = capacity_model(n)
    n_c = max(1, min(model_cap.n_controllers, n))

    features = node_features(
        graph,
        include_betweenness=include_betweenness,
        betweenness_k=betweenness_k,
        seed=episode_seed,
    )
    x = torch.tensor(features, dtype=torch.float32, device=device)
    edge_index = edge_index_from_graph(graph).to(device)

    with torch.no_grad():
        h = model.encode(x, edge_index)
        base_logits = model.logits(h)
        value = float(model.value(h))

    generator = _rng.derive(episode_seed, "ppo", "rollout")
    used: set[int] = set()
    steps: list[tuple[tuple[int, ...], int, float, float]] = []
    selected: list[int] = []
    for _ in range(n_c):
        probs, log_probs, entropy = masked_distribution(base_logits, used)
        probs_np = probs.detach().cpu().numpy()
        cum = np.cumsum(probs_np)
        draw = float(generator.random()) * float(cum[-1])
        pick = int(np.searchsorted(cum, draw, side="right"))
        pick = min(pick, n - 1)
        steps.append((tuple(sorted(used)), pick, float(log_probs[pick]), float(entropy)))
        used.add(pick)
        selected.append(pick)

    controllers = sorted(selected)
    reward = float(placement_reward(graph, controllers, cfg, seed=episode_seed)["weighted"])
    returns = compute_returns([0.0] * (len(steps) - 1) + [reward], gamma=cfg.gamma)

    return {
        "x": x,
        "edge_index": edge_index,
        "steps": steps,
        "value": value,
        "reward": reward,
        "returns": returns,
        "controllers": controllers,
        "n": n,
    }


def _ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    episodes: list[dict],
    cfg: TrainConfig,
    *,
    value_coef: float = 0.5,
    grad_clip: float = 0.5,
    minibatch: int = 32,
) -> None:
    """Run ``cfg.epochs`` PPO epochs, **minibatched** over the collected episodes."""
    if not episodes:
        return
    batch_size = minibatch if minibatch and minibatch > 0 else len(episodes)
    for _ in range(cfg.epochs):
        for start in range(0, len(episodes), batch_size):
            batch = episodes[start:start + batch_size]
            losses: list[torch.Tensor] = []
            for episode in batch:
                h = model.encode(episode["x"], episode["edge_index"])
                base_logits = model.logits(h)
                value = model.value(h)

                new_log_probs: list[torch.Tensor] = []
                entropies: list[torch.Tensor] = []
                for used, pick, _old_logp, _old_ent in episode["steps"]:
                    _probs, log_probs, entropy = masked_distribution(base_logits, used)
                    new_log_probs.append(log_probs[pick])
                    entropies.append(entropy)
                new_log_probs_t = torch.stack(new_log_probs)
                entropy_t = torch.stack(entropies)

                old_log_probs = torch.tensor(
                    [step[2] for step in episode["steps"]], dtype=torch.float32
                )
                returns = torch.tensor(episode["returns"], dtype=torch.float32)
                advantages = returns - float(episode["value"])
                target_value = torch.tensor(float(episode["reward"]), dtype=torch.float32)

                loss, _pl, _vl, _ent = surrogate_loss(
                    new_log_probs_t,
                    old_log_probs,
                    advantages,
                    value,
                    target_value,
                    entropy_t,
                    clip=cfg.clip,
                    value_coef=value_coef,
                    entropy_coef=cfg.entropy_coef,
                )
                losses.append(loss)
            batch_loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()


def _evaluate_val(
    model: ActorCritic,
    graphs: list[nx.Graph],
    cfg: TrainConfig,
    *,
    include_betweenness: bool,
    betweenness_k: int,
    device: str,
) -> float:
    """Mean terminal reward of a **greedy** placement on the (fixed) val subset."""
    if not graphs:
        return 0.0
    was_training = model.training
    model.eval()
    rewards: list[float] = []
    with torch.no_grad():
        for index, graph in enumerate(graphs):
            n = graph.number_of_nodes()
            n_c = max(1, min(capacity_model(n).n_controllers, n))
            features = node_features(
                graph, include_betweenness=include_betweenness,
                betweenness_k=betweenness_k, seed=index,
            )
            x = torch.tensor(features, dtype=torch.float32, device=device)
            edge_index = edge_index_from_graph(graph).to(device)
            base_logits = model.logits(model.encode(x, edge_index))
            used: set[int] = set()
            selected: list[int] = []
            for _ in range(n_c):
                probs, _lp, _ent = masked_distribution(base_logits, used)
                pick = int(torch.argmax(probs))
                used.add(pick)
                selected.append(pick)
            rewards.append(
                float(placement_reward(graph, sorted(selected), cfg, seed=index)["weighted"])
            )
    if was_training:
        model.train()
    return float(np.mean(rewards)) if rewards else 0.0


def train(
    train_graphs: Sequence[GraphLike],
    cfg: TrainConfig,
    out_path: str | Path | None = None,
    *,
    seed: int = 0,
    verbose: bool = True,
    include_betweenness: bool = True,
    betweenness_k: int = 64,
    heads: int = 2,
    device: str = "cpu",
    val_graphs: Sequence[GraphLike] | None = None,
    patience: int = 20,
    eval_sample: int = 20,
    value_coef: float = 0.5,
    grad_clip: float = 0.5,
    minibatch: int = 32,
) -> dict[str, Any]:
    """Train a PPO placement policy and (optionally) persist a checkpoint."""
    graphs = resolve_graphs(train_graphs)
    if not graphs:
        raise ValueError("train needs at least one graph")
    in_dim = feature_dim(include_betweenness)

    val_subset: list[nx.Graph] = []
    if val_graphs:
        val_all = resolve_graphs(val_graphs)
        if eval_sample and eval_sample < len(val_all):
            gen = _rng.derive(seed, "ppo", "val_subset")
            order = gen.permutation(len(val_all))[:eval_sample]
            val_subset = [val_all[int(i)] for i in order]
        else:
            val_subset = list(val_all)

    model = build_model(in_dim, cfg.hidden_dim, cfg.gcn_layers, heads=heads, seed=seed, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history: list[float] = []
    val_history: list[float] = []
    best_val = float("-inf")
    since_best = 0
    stopped_at = 0
    n_graphs = len(graphs)
    train_order = _train_order(seed, n_graphs)
    started = time.time()
    for iteration in range(cfg.total_iterations):
        episodes: list[dict[str, Any]] = []
        rewards: list[float] = []
        for ep in range(cfg.episodes_per_iter):
            graph = graphs[train_order[(iteration * cfg.episodes_per_iter + ep) % n_graphs]]
            episode_seed = _rng.derive_int(
                seed, "ppo", "episode", iteration, ep, low=0, high=2**31 - 1
            )
            episode = _rollout(
                model,
                graph,
                cfg,
                episode_seed,
                include_betweenness=include_betweenness,
                betweenness_k=betweenness_k,
                device=device,
            )
            episodes.append(episode)
            rewards.append(episode["reward"])
        _ppo_update(
            model, optimizer, episodes, cfg,
            value_coef=value_coef, grad_clip=grad_clip, minibatch=minibatch,
        )
        mean_reward = float(np.mean(rewards))
        history.append(mean_reward)
        stopped_at = iteration + 1

        if val_subset:
            val_reward = _evaluate_val(
                model, val_subset, cfg,
                include_betweenness=include_betweenness,
                betweenness_k=betweenness_k,
                device=device,
            )
            val_history.append(val_reward)
            if val_reward > best_val + 1e-9:
                best_val = val_reward
                since_best = 0
            else:
                since_best += 1
            if verbose:
                LOGGER.info(
                    "iter %d/%d: train=%.4f val=%.4f best_val=%.4f since_best=%d",
                    iteration + 1, cfg.total_iterations, mean_reward, val_reward,
                    best_val, since_best,
                )
            if patience and since_best >= patience:
                if verbose:
                    LOGGER.info("early stop: val did not improve for %d iterations (stopped at iter %d)", patience, stopped_at)
                break
        elif verbose:
            LOGGER.info(
                "iter %d/%d: mean_reward=%.4f (min=%.4f max=%.4f)",
                iteration + 1, cfg.total_iterations, mean_reward, min(rewards), max(rewards),
            )
    elapsed = time.time() - started

    model.eval()
    checkpoint: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "cfg": cfg.to_dict(),
        "feature_schema": feature_config(
            include_betweenness=include_betweenness, betweenness_k=betweenness_k
        ),
        "decode": {
            "include_betweenness": bool(include_betweenness),
            "betweenness_k": int(betweenness_k),
            "heads": int(heads),
        },
        "feature_dim": int(in_dim),
        "hidden_dim": int(cfg.hidden_dim),
        "gcn_layers": int(cfg.gcn_layers),
        "seed": int(seed),
        "history": history,
        "val_history": val_history,
        "best_val": (None if best_val == float("-inf") else float(best_val)),
        "stopped_iteration": int(stopped_at),
        "patience": int(patience),
        "value_coef": float(value_coef),
        "grad_clip": float(grad_clip),
        "minibatch": int(minibatch),
        "train_seconds": round(elapsed, 3),
        "reward_weights": dict(DEFAULT_REWARD_WEIGHTS),
    }

    if out_path is not None:
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, target)
        if verbose:
            LOGGER.info("checkpoint saved: %s (%.2fs)", target, elapsed)

    return checkpoint
