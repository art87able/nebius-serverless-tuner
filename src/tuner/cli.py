from __future__ import annotations
import argparse
import os
import sys
from tuner.config import Config, SearchSpace, Sweep
from tuner.loop import tune
from tuner.report import render
from tuner.bench import run as bench_run
from tuner.agent import propose
from tuner.deploy import deploy, teardown
from tuner import adapters

DEFAULT_SPACE = SearchSpace(dtypes=("auto", "bfloat16"),
                            max_num_seqs=(64, 256),
                            quantizations=(None, "fp8"))

ENDPOINT_NAME = "tuner-endpoint"


def make_deploy_fn():
    runner = adapters.SubprocessRunner()
    return lambda cfg: deploy(cfg, runner, name=ENDPOINT_NAME)


def make_bench_fn(sweep: Sweep):
    def bench_fn(url: str):
        client = adapters.HttpClient(url, os.environ["NEBIUS_API_KEY"],
                                     model=os.environ.get("ENDPOINT_MODEL", ""))
        return bench_run(client, sweep)
    return bench_fn


def make_agent_fn():
    return lambda hist, space: propose(hist, space, adapters.llm_generate)


def make_teardown_fn():
    runner = adapters.SubprocessRunner()
    return lambda: teardown(ENDPOINT_NAME, runner)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tuner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--model", required=True)
    t.add_argument("--max-iters", type=int, default=3)
    t.add_argument("--budget-usd", type=float, default=5.0)
    t.add_argument("--gpu-rate", type=float, required=True)
    t.add_argument("--gpu", default="a100")
    t.add_argument("--concurrency", type=int, default=8)
    t.add_argument("--requests", type=int, default=16)
    args = ap.parse_args(argv)

    sweep = Sweep(concurrency=args.concurrency, input_tokens=128,
                  output_tokens=128, n_requests=args.requests)
    try:
        result = tune(model=args.model, base_config=Config(model=args.model),
                      search_space=DEFAULT_SPACE, sweep=sweep, gpu_rate=args.gpu_rate,
                      deploy_fn=make_deploy_fn(), bench_fn=make_bench_fn(sweep),
                      agent_fn=make_agent_fn(), max_iters=args.max_iters,
                      budget_usd=args.budget_usd)
        print(render(result))
        return 0
    finally:
        # Always release the endpoint so it can't keep billing (no spend cap on the account).
        try:
            make_teardown_fn()()
        except Exception as exc:  # best-effort: never mask the run's outcome
            print(f"warning: endpoint teardown failed ({exc}); "
                  f"check 'nebius ai endpoint list' and delete '{ENDPOINT_NAME}' manually",
                  file=sys.stderr)
