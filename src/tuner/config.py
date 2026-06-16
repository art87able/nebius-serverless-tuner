from __future__ import annotations
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Config:
    model: str
    dtype: str = "auto"               # auto | float16 | bfloat16
    max_num_seqs: int = 256
    max_model_len: int = 4096
    quantization: str | None = None   # None | "fp8" | "awq"


@dataclass(frozen=True)
class RequestResult:
    ttft_ms: float
    total_ms: float
    output_tokens: int


@dataclass(frozen=True)
class Metrics:
    ttft_ms_median: float
    tpot_ms_median: float
    throughput_tok_s: float
    total_output_tokens: int
    wall_seconds: float
    concurrency: int
    n_requests: int


@dataclass(frozen=True)
class CostReport:
    gpu_rate_usd_hr: float
    total_cost_usd: float
    cost_per_1m_tok: float


@dataclass(frozen=True)
class Sweep:
    concurrency: int
    input_tokens: int
    output_tokens: int
    n_requests: int


@dataclass(frozen=True)
class SearchSpace:
    dtypes: tuple[str, ...]
    max_num_seqs: tuple[int, ...]
    quantizations: tuple[str | None, ...]


@dataclass(frozen=True)
class Iteration:
    config: Config
    metrics: Metrics
    cost: CostReport


@dataclass(frozen=True)
class NextConfig:
    config: Config
    rationale: str


@dataclass(frozen=True)
class Stop:
    rationale: str


Decision = Union[NextConfig, Stop]


@dataclass(frozen=True)
class Result:
    model: str
    iterations: list[Iteration]
    winning: Iteration
    stopped_reason: str
