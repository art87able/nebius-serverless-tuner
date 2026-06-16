from __future__ import annotations
from typing import Callable
from tuner.config import (Config, SearchSpace, Sweep, Iteration, Result, Stop, Decision)
from tuner.pricing import cost


def tune(*, model: str, base_config: Config, search_space: SearchSpace, sweep: Sweep,
         gpu_rate: float,
         deploy_fn: Callable[[Config], str],
         bench_fn: Callable[[str], "object"],
         agent_fn: Callable[[list[Iteration], SearchSpace], Decision],
         max_iters: int, budget_usd: float) -> Result:
    iterations: list[Iteration] = []
    spent = 0.0
    config = base_config
    reason = "max_iters reached"
    for _ in range(max_iters):
        url = deploy_fn(config)
        metrics = bench_fn(url)
        cr = cost(metrics, gpu_rate)
        spent += cr.total_cost_usd
        iterations.append(Iteration(config, metrics, cr))
        if spent >= budget_usd:
            reason = "budget reached"
            break
        decision = agent_fn(iterations, search_space)
        if isinstance(decision, Stop):
            reason = f"agent stop: {decision.rationale}"
            break
        config = decision.config
    winning = min(iterations, key=lambda it: it.cost.cost_per_1m_tok)
    return Result(model=model, iterations=iterations, winning=winning, stopped_reason=reason)
