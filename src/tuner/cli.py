from __future__ import annotations
import argparse
import os
import secrets
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


def make_deploy_fn(token: str, url_holder: dict | None = None):
    runner = adapters.SubprocessRunner()
    subnet_id = adapters.get_subnet_id(runner)

    def _deploy(cfg):
        # The loop redeploys per config but only tears down once at the very end, so
        # clear any prior endpoint of this name first (no-op on the first iteration)
        # to avoid a name collision / leaked endpoint on iteration 2+.
        try:
            teardown(ENDPOINT_NAME, runner)
        except Exception:
            pass
        url = deploy(cfg, runner, token=token, subnet_id=subnet_id, name=ENDPOINT_NAME)
        # create returns before the GPU is serving; wait for vLLM to come up.
        adapters.wait_for_ready(url, token)
        if url_holder is not None:
            url_holder["url"] = url
        return url

    return _deploy


def make_bench_fn(sweep: Sweep, token: str):
    def bench_fn(url: str):
        client = adapters.HttpClient(url, token,
                                     model=os.environ.get("ENDPOINT_MODEL", ""))
        return bench_run(client, sweep)
    return bench_fn


def make_agent_fn(token: str | None = None, url_holder: dict | None = None,
                  model: str | None = None):
    # Self-contained mode: AGENT_LLM_BASE_URL=endpoint makes the tuner's own deployed
    # endpoint the agent's brain — no external LLM API needed. The loop calls the agent
    # right after benchmarking, while the current endpoint is still serving.
    if os.environ.get("AGENT_LLM_BASE_URL") == "endpoint" and url_holder is not None:
        generate = adapters.endpoint_llm_generate(
            lambda: url_holder.get("url"), token or "",
            os.environ.get("ENDPOINT_MODEL") or model or "")
        return lambda hist, space: propose(hist, space, generate)
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
    token = secrets.token_hex(32)   # the endpoint's bearer, shared with the benchmark client
    url_holder: dict = {"url": None}   # current endpoint URL, for endpoint-as-agent-brain mode
    try:
        result = tune(model=args.model, base_config=Config(model=args.model),
                      search_space=DEFAULT_SPACE, sweep=sweep, gpu_rate=args.gpu_rate,
                      deploy_fn=make_deploy_fn(token, url_holder),
                      bench_fn=make_bench_fn(sweep, token),
                      agent_fn=make_agent_fn(token, url_holder, args.model), max_iters=args.max_iters,
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
