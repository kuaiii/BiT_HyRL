"""Configuration dataclasses with validation, ``to_dict`` / ``from_dict``.

Every knob that reaches the pipeline is declared **once**, here, as a frozen
dataclass.  Rationale:

* a frozen dataclass cannot be mutated halfway through a run, so a record's
  configuration can never disagree with the code that produced it;
* validation lives in ``__post_init__``, so an invalid value fails at
  construction time rather than 40 minutes into a sweep;
* ``from_dict(to_dict(x)) == x`` is a round-trip identity, which lets the whole
  configuration travel through JSON without a second, hand-written parser.

Configuration is **JSON, not YAML** — the runtime environment used for this
project has no ``pyyaml`` installed.  ``json`` from the standard
library is therefore the only serialisation format used.
"""

from __future__ import annotations

import math
import types
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Mapping

__all__ = [
    "ConfigError",
    "TopologyConfig",
    "CapacityConfig",
    "CascadeConfig",
    "MetricsConfig",
    "AttackConfig",
    "TrainConfig",
    "TransferConfig",
    "PipelineConfig",
    "CASCADE_MODES",
    "CAPACITY_MODES",
    "REASSIGN_POLICIES",
    "ATTACK_METHODS",
    "CONTROL_POLICIES",
    "REWARD_WEIGHT_KEYS",
]

#: Supported cascade semantics.
CASCADE_MODES = ("structural", "capacity")

#: Supported capacity derivations.
CAPACITY_MODES = ("formula", "explicit")

#: Supported overflow re-assignment policies.
REASSIGN_POLICIES = ("nearest_with_capacity", "nearest")

#: Tier-A attack methods implemented in-package (Tier-B lives behind a bridge).
ATTACK_METHODS = (
    "random",
    "degree",
    "betweenness",
    "closeness",
    "coreness",
    "eigenvector",
)

#: Baseline controller-placement policies (the PPO policy is configured by
#: ``TrainConfig`` / ``TransferConfig`` instead).
CONTROL_POLICIES = ("random", "degree", "betweenness", "kmedian", "ppo")

#: Exactly the three decomposition blocks plus the distance term.
REWARD_WEIGHT_KEYS = ("structure", "csa", "h_load", "distance")


class ConfigError(ValueError):
    """Raised when a configuration object violates its own contract."""


# --------------------------------------------------------------------------- #
# generic (de)serialisation
# --------------------------------------------------------------------------- #
def _plain(value: Any) -> Any:
    """Recursively convert a value into JSON-friendly plain Python types."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _coerce(value: Any, hint: Any, where: str) -> Any:
    """Coerce ``value`` to the type described by ``hint``.

    Args:
        value: Raw value coming from JSON.
        hint: Resolved type annotation of the target field.
        where: Dotted path used in error messages.

    Returns:
        The coerced value.

    Raises:
        ConfigError: If the value cannot be coerced without loss.
    """
    if hint is Any or hint is None:
        return value
    # A checkpoint or a JSON round-trip can hand back a numpy scalar (``np.int64``)
    # where a Python scalar was written.  numpy scalars are not instances of ``int``
    # or ``float``, so the exact-type branches below would reject a value that is
    # numerically identical; normalise them here instead of loosening those branches.
    if type(value).__module__ == "numpy":
        item = getattr(value, "item", None)
        if item is not None:
            value = item()
    if hint is type(None):
        if value is not None:
            raise ConfigError(f"{where}: 期望 null，收到 {value!r}")
        return None

    origin = typing.get_origin(hint)
    args = typing.get_args(hint)

    # ``types.UnionType`` only exists from 3.10; asking for it with a default of
    # ``None`` makes the comparison ``None is None`` on 3.9, which silently routes
    # every plain scalar hint through the union branch.  Guard the second half.
    union_type = getattr(types, "UnionType", None)
    if origin is typing.Union or (union_type is not None and origin is union_type):
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{where}: 不允许 null")
        candidates = [a for a in args if a is not type(None)]
        if len(candidates) == 1:
            return _coerce(value, candidates[0], where)
        last: Exception | None = None
        for candidate in candidates:
            try:
                return _coerce(value, candidate, where)
            except ConfigError as exc:  # pragma: no cover - defensive
                last = exc
        raise ConfigError(f"{where}: 无法匹配任何联合类型分支") from last

    if origin in (tuple, list) or hint in (tuple, list):
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: 期望序列，收到 {type(value).__name__}")
        items = list(value)
        if not args:  # bare tuple / list
            return tuple(items) if hint in (tuple, list) or origin is tuple else items
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0], f"{where}[{i}]") for i, v in enumerate(items))
        if len(args) == 1:
            return tuple(_coerce(v, args[0], f"{where}[{i}]") for i, v in enumerate(items))
        if len(args) != len(items):
            raise ConfigError(f"{where}: 期望 {len(args)} 个元素，收到 {len(items)}")
        return tuple(_coerce(v, a, f"{where}[{i}]") for i, (v, a) in enumerate(zip(items, args)))

    if origin is dict or hint is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: 期望对象，收到 {type(value).__name__}")
        if len(args) == 2:
            return {
                _coerce(k, args[0], f"{where}.key"): _coerce(v, args[1], f"{where}[{k!r}]")
                for k, v in value.items()
            }
        return dict(value)

    if hint is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{where}: 期望布尔值，收到 {value!r}")

    if hint is int:
        if isinstance(value, bool):
            raise ConfigError(f"{where}: 布尔值不能当整数用")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and float(value).is_integer():
            return int(value)
        raise ConfigError(f"{where}: 期望整数，收到 {value!r}")

    if hint is float:
        if isinstance(value, bool):
            raise ConfigError(f"{where}: 布尔值不能当浮点数用")
        if isinstance(value, (int, float)):
            return float(value)
        raise ConfigError(f"{where}: 期望浮点数，收到 {value!r}")

    if hint is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"{where}: 期望字符串，收到 {value!r}")

    if isinstance(hint, type) and is_dataclass(hint):
        if isinstance(value, hint):
            return value
        if isinstance(value, Mapping) and hasattr(hint, "from_dict"):
            return hint.from_dict(value)
        raise ConfigError(f"{where}: 期望 {hint.__name__} 配置块，收到 {type(value).__name__}")

    return value


class _ConfigBase:
    """Shared ``to_dict`` / ``from_dict`` behaviour for frozen config objects."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable plain-``dict`` view of this config."""
        return _plain(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "_ConfigBase":
        """Build an instance from a mapping, validating every key.

        Args:
            data: Mapping as produced by :meth:`to_dict` (or by ``json.load``).

        Returns:
            A new instance.

        Raises:
            ConfigError: On unknown keys, missing required values, or values
                that violate the field type or the class contract.
        """
        if not isinstance(data, Mapping):
            raise ConfigError(f"{cls.__name__}.from_dict 需要映射，收到 {type(data).__name__}")
        hints = typing.get_type_hints(cls)
        known = {f.name for f in fields(cls)}
        # Keys starting with "_" are documentation-only (``_comment``), the usual
        # convention for comments in JSON; everything else must be a real field so
        # a typo cannot silently fall back to a default.
        payload = {key: value for key, value in data.items() if not str(key).startswith("_")}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ConfigError(f"{cls.__name__}: 未知配置项 {unknown}")
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in payload:
                continue
            kwargs[f.name] = _coerce(payload[f.name], hints.get(f.name, Any), f"{cls.__name__}.{f.name}")
        return cls(**kwargs)

    def validate(self) -> None:
        """Re-run the contract checks (no-op hook; concrete classes override)."""


def _require_finite(name: str, value: float, *, low: float | None = None,
                    high: float | None = None, low_open: bool = False,
                    high_open: bool = False) -> None:
    """Validate a float knob against a finite interval."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{name} 必须是数值，收到 {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{name} 必须有限，收到 {number!r}")
    if low is not None:
        if low_open and number <= low:
            raise ConfigError(f"{name} 必须 > {low}，收到 {number}")
        if not low_open and number < low:
            raise ConfigError(f"{name} 必须 >= {low}，收到 {number}")
    if high is not None:
        if high_open and number >= high:
            raise ConfigError(f"{name} 必须 < {high}，收到 {number}")
        if not high_open and number > high:
            raise ConfigError(f"{name} 必须 <= {high}，收到 {number}")


def _require_int(name: str, value: int, *, low: int | None = None,
                 high: int | None = None) -> None:
    """Validate an integer knob against an interval."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} 必须是整数，收到 {value!r}")
    if low is not None and value < low:
        raise ConfigError(f"{name} 必须 >= {low}，收到 {value}")
    if high is not None and value > high:
        raise ConfigError(f"{name} 必须 <= {high}，收到 {value}")


# --------------------------------------------------------------------------- #
# BiT topology layer
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TopologyConfig(_ConfigBase):
    """BiT construction knobs.

    ``c`` is the *only* shape parameter of the bimodal construction: the degree
    sequence is fully determined by ``(n, m, c, h)``, and ``k_max`` is
    ``round_half_up(c * k_bar)``.  ``c -> 1`` gives a degree-balanced graph,
    ``c -> large`` approaches a single super-hub.

    Attributes:
        c: Hub factor. Default ``1.6`` (the headline configuration).
        h: Number of hub nodes. Default ``1``.
        enforce_connected: If True, repair disconnection with degree-preserving
            2-switches and raise when the repair budget is exhausted.
        max_hh_attempts: Maximum number of Havel-Hakimi downgrade attempts.
        connect_repair_swaps: Maximum number of 2-switches spent on
            connectivity repair.
    """

    c: float = 1.6
    h: int = 1
    enforce_connected: bool = True
    max_hh_attempts: int = 64
    connect_repair_swaps: int = 20000

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check that every knob is inside its admissible range."""
        _require_finite("TopologyConfig.c", self.c, low=0.0, low_open=True)
        _require_int("TopologyConfig.h", self.h, low=1)
        if not isinstance(self.enforce_connected, bool):
            raise ConfigError("TopologyConfig.enforce_connected 必须是布尔值")
        _require_int("TopologyConfig.max_hh_attempts", self.max_hh_attempts, low=1)
        _require_int("TopologyConfig.connect_repair_swaps", self.connect_repair_swaps, low=0)


# --------------------------------------------------------------------------- #
# control plane
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CapacityConfig(_ConfigBase):
    """Controller capacity model.

    With the defaults ``theta=0.5`` and ``controller_ratio=0.1`` the per
    controller capacity is ``ceil(1.5 * 10) = 15`` for *every* ``N``, because
    ``N / n_c == 10`` exactly.  Aggregate capacity is then ``1.5 N`` against a
    demand of ``0.9 N``: 40 % slack, so whether capacity binds is a question
    about the *spatial* load distribution, not about the totals.

    Attributes:
        theta: Capacity slack factor.
        controller_ratio: ``n_c = round_half_up(controller_ratio * N)``.
        n_controllers: Explicit override; wins over ``controller_ratio``.
        demand: Uniform demand ``l_i(t)``.
        capacity_mode: ``"formula"`` or ``"explicit"``.
        capacity_explicit: Capacity value used when ``capacity_mode ==
            "explicit"``.
    """

    theta: float = 0.5
    controller_ratio: float = 0.1
    n_controllers: int | None = None
    demand: int = 1
    capacity_mode: str = "formula"
    capacity_explicit: int | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check theoretical/structural consistency of the capacity model."""
        _require_finite("CapacityConfig.theta", self.theta, low=0.0)
        _require_finite("CapacityConfig.controller_ratio", self.controller_ratio,
                        low=0.0, high=1.0, low_open=True)
        if self.n_controllers is not None:
            _require_int("CapacityConfig.n_controllers", self.n_controllers, low=1)
        _require_int("CapacityConfig.demand", self.demand, low=1)
        if self.capacity_mode not in CAPACITY_MODES:
            raise ConfigError(
                f"CapacityConfig.capacity_mode 必须是 {CAPACITY_MODES}，收到 {self.capacity_mode!r}"
            )
        if self.capacity_mode == "explicit":
            if self.capacity_explicit is None:
                raise ConfigError("capacity_mode='explicit' 时必须给出 capacity_explicit")
            _require_int("CapacityConfig.capacity_explicit", self.capacity_explicit, low=1)
        elif self.capacity_explicit is not None:
            raise ConfigError("capacity_mode='formula' 时不允许设置 capacity_explicit")


@dataclass(frozen=True)
class CascadeConfig(_ConfigBase):
    """Cascade semantics.

    ``structural`` is the default because it is the semantics behind the
    already-published results; ``capacity`` exists only for the appendix
    sensitivity analysis.

    Attributes:
        mode: ``"structural"`` (default) or ``"capacity"``.
        max_iter: Fixed-point iteration cap for the capacity mode.
        reassign_policy: Overflow re-assignment rule.
    """

    mode: str = "structural"
    max_iter: int = 50
    reassign_policy: str = "nearest_with_capacity"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check cascade mode and iteration budget."""
        if self.mode not in CASCADE_MODES:
            raise ConfigError(f"CascadeConfig.mode 必须是 {CASCADE_MODES}，收到 {self.mode!r}")
        _require_int("CascadeConfig.max_iter", self.max_iter, low=1, high=10000)
        if self.reassign_policy not in REASSIGN_POLICIES:
            raise ConfigError(
                f"CascadeConfig.reassign_policy 必须是 {REASSIGN_POLICIES}，"
                f"收到 {self.reassign_policy!r}"
            )


@dataclass(frozen=True)
class MetricsConfig(_ConfigBase):
    """Metric definitions.

    Attributes:
        alpha: CSA attenuation factor (``1 - exp(-alpha * k_i)``).
        collapse_gcc_threshold: GCC fraction below which the graph counts as
            collapsed.
        half_window: Upper bound of the half-window AUC integral.
        epsilon: Numerical tolerance for "is this float zero".
    """

    alpha: float = 0.2
    collapse_gcc_threshold: float = 0.2
    half_window: float = 0.5
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check metric parameters."""
        _require_finite("MetricsConfig.alpha", self.alpha, low=0.0, high=1.0, low_open=True)
        _require_finite("MetricsConfig.collapse_gcc_threshold",
                        self.collapse_gcc_threshold, low=0.0, high=1.0, low_open=True)
        _require_finite("MetricsConfig.half_window", self.half_window,
                        low=0.0, high=1.0, low_open=True)
        _require_finite("MetricsConfig.epsilon", self.epsilon, low=0.0, low_open=True)


# --------------------------------------------------------------------------- #
# attacks, training, transfer
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AttackConfig(_ConfigBase):
    """Attack protocol selection.

    Attributes:
        method: Tier-A method name (see :data:`ATTACK_METHODS`).
        seed: Root seed for the attack stream.
        multistage: Whether to compose methods across the three ``f`` segments.
        reinsertion: Whether the protocol applies re-insertion.
        params: Method-specific parameters.
    """

    method: str = "degree"
    seed: int = 0
    multistage: bool = False
    reinsertion: bool = False
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check method name, seed domain and parameter container."""
        if not isinstance(self.method, str) or not self.method.strip():
            raise ConfigError("AttackConfig.method 必须是非空字符串")
        _require_int("AttackConfig.seed", self.seed, low=0, high=2**32 - 1)
        if not isinstance(self.multistage, bool) or not isinstance(self.reinsertion, bool):
            raise ConfigError("AttackConfig.multistage / reinsertion 必须是布尔值")
        if not isinstance(self.params, Mapping):
            raise ConfigError("AttackConfig.params 必须是映射")


@dataclass(frozen=True)
class TrainConfig(_ConfigBase):
    """PPO training hyper-parameters.

    Attributes:
        total_iterations: Outer PPO iterations.
        episodes_per_iter: Rollout episodes per iteration.
        lr: Adam learning rate.
        clip: PPO clipping range.
        gae_lambda: GAE smoothing factor.
        gamma: Discount factor (1.0 = no discounting, as the horizon is fixed).
        entropy_coef: Entropy bonus weight.
        epochs: Optimisation epochs per rollout.
        hidden_dim: GCN hidden width.
        gcn_layers: Number of graph-convolution layers.
        reward_weights: Weights of the four reward blocks; the key set is fixed
            so that a reward can never silently lose a term.
        attack_prefix_fraction: Fraction of the removal sequence used inside the
            reward (short-prefix evaluation).
    """

    total_iterations: int = 200
    episodes_per_iter: int = 8
    lr: float = 3e-4
    clip: float = 0.2
    gae_lambda: float = 0.95
    gamma: float = 1.0
    entropy_coef: float = 0.01
    epochs: int = 4
    hidden_dim: int = 64
    gcn_layers: int = 2
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {"structure": 1.0, "csa": 0.5, "h_load": 0.5, "distance": 0.1}
    )
    attack_prefix_fraction: float = 0.3

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check every hyper-parameter and the reward-weight key set."""
        _require_int("TrainConfig.total_iterations", self.total_iterations, low=1)
        _require_int("TrainConfig.episodes_per_iter", self.episodes_per_iter, low=1)
        _require_finite("TrainConfig.lr", self.lr, low=0.0, low_open=True)
        _require_finite("TrainConfig.clip", self.clip, low=0.0, high=1.0, low_open=True)
        _require_finite("TrainConfig.gae_lambda", self.gae_lambda, low=0.0, high=1.0)
        _require_finite("TrainConfig.gamma", self.gamma, low=0.0, high=1.0, low_open=True)
        _require_finite("TrainConfig.entropy_coef", self.entropy_coef, low=0.0)
        _require_int("TrainConfig.epochs", self.epochs, low=1)
        _require_int("TrainConfig.hidden_dim", self.hidden_dim, low=1)
        _require_int("TrainConfig.gcn_layers", self.gcn_layers, low=1)
        if not isinstance(self.reward_weights, Mapping):
            raise ConfigError("TrainConfig.reward_weights 必须是映射")
        keys = set(self.reward_weights)
        expected = set(REWARD_WEIGHT_KEYS)
        if keys != expected:
            raise ConfigError(
                f"TrainConfig.reward_weights 的键必须恰好是 {sorted(expected)}，"
                f"缺失 {sorted(expected - keys)}，多余 {sorted(keys - expected)}"
            )
        for key in REWARD_WEIGHT_KEYS:
            _require_finite(f"TrainConfig.reward_weights[{key}]",
                            self.reward_weights[key], low=0.0)
        _require_finite("TrainConfig.attack_prefix_fraction",
                        self.attack_prefix_fraction, low=0.0, high=1.0, low_open=True)


@dataclass(frozen=True)
class TransferConfig(_ConfigBase):
    """Zero-shot transfer configuration.

    ``per_graph_retrain`` is an *explicit* switch whose default is ``False``.
    Mixed reporting of per-graph retraining and zero-shot transfer in a single
    table is the exact failure mode identified during revision, so the
    flag is mandatory in every record.

    Attributes:
        per_graph_retrain: If True, retrain on each target graph.
        checkpoint: Path to the source checkpoint.
        source_datasets: Datasets the checkpoint was trained on.
    """

    per_graph_retrain: bool = False
    checkpoint: str | None = None
    source_datasets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check the explicit retrain flag and checkpoint typing."""
        if not isinstance(self.per_graph_retrain, bool):
            raise ConfigError("TransferConfig.per_graph_retrain 必须是布尔值")
        if self.checkpoint is not None and not isinstance(self.checkpoint, str):
            raise ConfigError("TransferConfig.checkpoint 必须是字符串或 null")
        if not isinstance(self.source_datasets, tuple):
            raise ConfigError("TransferConfig.source_datasets 必须是元组")
        for item in self.source_datasets:
            if not isinstance(item, str):
                raise ConfigError("TransferConfig.source_datasets 只能包含字符串")


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PipelineConfig(_ConfigBase):
    """The whole configuration tree (what ``configs/default.json`` holds)."""

    topology: TopologyConfig = field(default_factory=TopologyConfig)
    capacity: CapacityConfig = field(default_factory=CapacityConfig)
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Re-validate every nested block."""
        for block in (self.topology, self.capacity, self.cascade, self.metrics,
                      self.attack, self.train, self.transfer):
            block.validate()
