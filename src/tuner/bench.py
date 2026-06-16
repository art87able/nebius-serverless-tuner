from __future__ import annotations
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol
from tuner.config import RequestResult, Metrics, Sweep


class Client(Protocol):
    def chat(self, input_tokens: int, output_tokens: int) -> RequestResult: ...


def aggregate(results: list[RequestResult], wall_seconds: float, concurrency: int) -> Metrics:
    ttfts = [r.ttft_ms for r in results]
    tpots = [(r.total_ms - r.ttft_ms) / max(r.output_tokens - 1, 1) for r in results]
    total_tokens = sum(r.output_tokens for r in results)
    throughput = total_tokens / wall_seconds if wall_seconds else 0.0
    return Metrics(
        ttft_ms_median=round(statistics.median(ttfts), 2),
        tpot_ms_median=round(statistics.median(tpots), 2),
        throughput_tok_s=round(throughput, 2),
        total_output_tokens=total_tokens,
        wall_seconds=round(wall_seconds, 3),
        concurrency=concurrency,
        n_requests=len(results),
    )


def run(client: "Client", sweep: Sweep, time_fn=time.monotonic) -> Metrics:
    start = time_fn()
    if sweep.concurrency <= 1:
        results = [client.chat(sweep.input_tokens, sweep.output_tokens)
                   for _ in range(sweep.n_requests)]
    else:
        with ThreadPoolExecutor(max_workers=sweep.concurrency) as ex:
            futs = [ex.submit(client.chat, sweep.input_tokens, sweep.output_tokens)
                    for _ in range(sweep.n_requests)]
            results = [f.result() for f in futs]
    wall = time_fn() - start
    return aggregate(results, wall_seconds=wall, concurrency=sweep.concurrency)
