from tuner.config import Config, Metrics, CostReport, Iteration, Result
from tuner.report import render


def _it(quant, cost_1m):
    return Iteration(Config(model="m", quantization=quant),
                     Metrics(50, 10, 500, 1000, 1.0, 8, 16),
                     CostReport(2.0, 0.01, cost_1m))


def test_render_contains_table_and_winner():
    it1, it2 = _it(None, 2.0), _it("fp8", 1.2)
    r = Result(model="Qwen/Qwen2.5-1.5B-Instruct", iterations=[it1, it2],
               winning=it2, stopped_reason="agent stop: converged")
    md = render(r)
    assert "Qwen/Qwen2.5-1.5B-Instruct" in md
    assert "cost/1M" in md
    assert "1.2" in md                       # winning cost appears
    assert "Winning config" in md
    assert "fp8" in md
    assert "agent stop: converged" in md
