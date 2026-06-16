from tuner.config import Metrics
from tuner.pricing import cost


def _metrics(total_tokens, wall_seconds):
    return Metrics(0, 0, 0, total_tokens, wall_seconds, 1, 1)


def test_cost_per_million_tokens():
    # 1 GPU-hour at $2/hr produces 1,000,000 tokens => $2 per 1M tok
    m = _metrics(total_tokens=1_000_000, wall_seconds=3600)
    cr = cost(m, gpu_rate_usd_hr=2.0)
    assert cr.total_cost_usd == 2.0
    assert cr.cost_per_1m_tok == 2.0


def test_zero_tokens_is_inf():
    cr = cost(_metrics(0, 10), gpu_rate_usd_hr=2.0)
    assert cr.cost_per_1m_tok == float("inf")
