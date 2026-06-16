from tuner.config import (
    Config, RequestResult, Metrics, CostReport, Sweep,
    SearchSpace, Iteration, NextConfig, Stop, Result,
)


def test_config_defaults():
    c = Config(model="Qwen/Qwen2.5-1.5B-Instruct")
    assert c.dtype == "auto"
    assert c.max_num_seqs == 256
    assert c.quantization is None


def test_result_holds_winning_iteration():
    cfg = Config(model="m")
    m = Metrics(1.0, 1.0, 1.0, 10, 1.0, 1, 1)
    it = Iteration(cfg, m, CostReport(1.0, 0.1, 0.5))
    r = Result(model="m", iterations=[it], winning=it, stopped_reason="x")
    assert r.winning is it
