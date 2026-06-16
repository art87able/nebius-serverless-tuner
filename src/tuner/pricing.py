from __future__ import annotations
from tuner.config import Metrics, CostReport

# GPU hourly rates (USD). VERIFY against live Nebius pricing during the live pass;
# `cost()` itself is rate-agnostic -- callers pass the rate for the GPU they deployed on.
NEBIUS_GPU_RATES_USD_HR = {
    "l40s": 1.0,
    "a100": 2.0,
    "h100": 3.0,
}


def cost(metrics: Metrics, gpu_rate_usd_hr: float) -> CostReport:
    total = metrics.wall_seconds / 3600 * gpu_rate_usd_hr
    if metrics.total_output_tokens:
        per_1m = total / (metrics.total_output_tokens / 1_000_000)
    else:
        per_1m = float("inf")
    return CostReport(
        gpu_rate_usd_hr=gpu_rate_usd_hr,
        total_cost_usd=round(total, 6),
        cost_per_1m_tok=round(per_1m, 4) if per_1m != float("inf") else float("inf"),
    )
