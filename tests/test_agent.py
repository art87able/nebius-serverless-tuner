from tuner.config import Config, Metrics, CostReport, Iteration, SearchSpace, NextConfig, Stop
from tuner.agent import propose

SPACE = SearchSpace(dtypes=("auto", "bfloat16"),
                    max_num_seqs=(64, 256),
                    quantizations=(None, "fp8"))


def _iter(cfg):
    m = Metrics(50, 10, 100, 1000, 1.0, 1, 1)
    return Iteration(cfg, m, CostReport(2.0, 0.001, 2.0))


def test_propose_returns_next_config():
    history = [_iter(Config(model="m"))]
    gen = lambda p: '{"action":"continue","dtype":"bfloat16","max_num_seqs":64,"quantization":"fp8","rationale":"try fp8"}'
    d = propose(history, SPACE, gen)
    assert isinstance(d, NextConfig)
    assert d.config.quantization == "fp8"
    assert d.config.max_num_seqs == 64
    assert d.config.model == "m"          # model is held fixed


def test_propose_returns_stop():
    history = [_iter(Config(model="m"))]
    gen = lambda p: '{"action":"stop","rationale":"converged"}'
    d = propose(history, SPACE, gen)
    assert isinstance(d, Stop)


def test_propose_out_of_space_stops():
    history = [_iter(Config(model="m"))]
    gen = lambda p: '{"action":"continue","dtype":"int4","max_num_seqs":99999,"rationale":"x"}'
    d = propose(history, SPACE, gen)
    assert isinstance(d, Stop)
