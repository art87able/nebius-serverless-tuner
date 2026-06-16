from tuner.config import RequestResult
from tuner.bench import aggregate


def test_aggregate_basic():
    # two requests: ttft 100/200; each 11 output tokens over (total-ttft)=100ms
    # tpot = 100 / (11-1) = 10 ms/token
    results = [
        RequestResult(ttft_ms=100, total_ms=200, output_tokens=11),
        RequestResult(ttft_ms=200, total_ms=300, output_tokens=11),
    ]
    m = aggregate(results, wall_seconds=2.0, concurrency=2)
    assert m.ttft_ms_median == 150.0       # median of 100,200
    assert m.tpot_ms_median == 10.0
    assert m.total_output_tokens == 22
    assert m.throughput_tok_s == 11.0      # 22 tokens / 2.0 s
    assert m.concurrency == 2
    assert m.n_requests == 2


from tuner.bench import run, Client  # noqa: E402


class FakeClient:
    def __init__(self):
        self.calls = 0

    def chat(self, input_tokens: int, output_tokens: int) -> RequestResult:
        self.calls += 1
        return RequestResult(ttft_ms=100, total_ms=200, output_tokens=output_tokens)


def test_run_issues_n_requests_and_aggregates():
    from tuner.config import Sweep
    client = FakeClient()
    clock = iter([0.0, 4.0])  # start, end -> wall 4.0s
    sweep = Sweep(concurrency=1, input_tokens=32, output_tokens=11, n_requests=4)
    m = run(client, sweep, time_fn=lambda: next(clock))
    assert client.calls == 4
    assert m.n_requests == 4
    assert m.wall_seconds == 4.0
    assert m.total_output_tokens == 44   # 4 * 11
