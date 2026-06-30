from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence, Tuple, Union

SpecMode = Literal["gue", "semicircle", "poisson", "picket"]
LambdaDope = Union[int, Literal["haar"]]


@dataclass(frozen=True)
class AvalancheConfig:
    N0: int = 10
    L: int = 10
    xi: float = 2.0
    g: float = 1.0
    h_dist: Tuple[str, float, float] = ("uniform", -1.0, 1.0)
    J: float = 0.0
    lambda_dope: LambdaDope = 0
    spec_mode: SpecMode = "gue"
    W: float = 1.0
    dt: float = 0.1
    t_max: float = 2.0
    n_typicality: int = 4
    seed: int = 0
    extra: Tuple[Any, ...] = field(default_factory=tuple)

    @property
    def dim_bath(self) -> int:
        return 1 << self.N0

    @property
    def dim_spins(self) -> int:
        return 1 << self.L

    @property
    def dim_total(self) -> int:
        return self.dim_bath * self.dim_spins

    @property
    def param_hash(self) -> str:
        values = {
            k: v
            for k, v in asdict(self).items()
            if k not in {"seed", "extra"}
        }
        normalized = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def with_seed(self, seed: int) -> "AvalancheConfig":
        return AvalancheConfig(**{**asdict(self), "seed": seed})
