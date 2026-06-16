from tuner.config import (Config, Metrics, SearchSpace, NextConfig, Stop, Sweep)
from tuner.loop import tune

SPACE = SearchSpace(("auto",), (64, 256), (None, "fp8"))
SWEEP = Sweep(concurrency=1, input_tokens=32, output_tokens=10, n_requests=2)


def _metrics(tput):
    # tokens scale with throughput at fixed wall => higher tput = cheaper per token
    return Metrics(50, 10, tput, tput, 1.0, 1, 2)


def test_tune_picks_lowest_cost_and_respects_max_iters():
    # bench returns improving throughput so later configs are cheaper
    seq = iter([_metrics(100), _metrics(500), _metrics(900)])
    deploy_fn = lambda cfg: "https://e/v1"
    bench_fn = lambda url: next(seq)
    # agent always proposes another config; max_iters caps it at 2
    agent_fn = lambda hist, space: NextConfig(Config(model="m", max_num_seqs=256), "go")
    r = tune(model="m", base_config=Config(model="m"), search_space=SPACE, sweep=SWEEP,
             gpu_rate=2.0, deploy_fn=deploy_fn, bench_fn=bench_fn, agent_fn=agent_fn,
             max_iters=2, budget_usd=1000.0)
    assert len(r.iterations) == 2
    assert r.stopped_reason == "max_iters reached"
    # higher throughput => lower cost/1M => that iteration wins
    assert r.winning.metrics.throughput_tok_s == 500


def test_tune_stops_on_agent_stop():
    deploy_fn = lambda cfg: "https://e/v1"
    bench_fn = lambda url: _metrics(100)
    agent_fn = lambda hist, space: Stop("converged")
    r = tune(model="m", base_config=Config(model="m"), search_space=SPACE, sweep=SWEEP,
             gpu_rate=2.0, deploy_fn=deploy_fn, bench_fn=bench_fn, agent_fn=agent_fn,
             max_iters=5, budget_usd=1000.0)
    assert len(r.iterations) == 1
    assert "agent stop" in r.stopped_reason


def test_tune_stops_on_budget():
    deploy_fn = lambda cfg: "https://e/v1"
    # wall_seconds=3600 at $2/hr => $2/iteration; budget 1.5 stops after first
    bench_fn = lambda url: Metrics(50, 10, 100, 1000, 3600, 1, 2)
    agent_fn = lambda hist, space: NextConfig(Config(model="m"), "go")
    r = tune(model="m", base_config=Config(model="m"), search_space=SPACE, sweep=SWEEP,
             gpu_rate=2.0, deploy_fn=deploy_fn, bench_fn=bench_fn, agent_fn=agent_fn,
             max_iters=5, budget_usd=1.5)
    assert r.stopped_reason == "budget reached"
    assert len(r.iterations) == 1
