# Serverless Tuner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an agentic AI-performance-engineering tool that deploys an open model as a Nebius Serverless Endpoint, benchmarks it, and autonomously tunes the serving config to minimise cost-per-token — emitting an honest report. Submission for the Nebius Serverless AI Builders Challenge.

**Architecture:** Six pure-ish units behind injected seams (`CliRunner`, `Client`, `Generate`) so the whole deploy→bench→agent→report loop runs fully mocked in CI with zero network and zero GPU. Floor = single-config cost/perf report; ceiling = the bounded autonomous tune loop.

**Tech Stack:** Python 3.11+, `pytest`, `requests` (real HTTP client), `openai`/`requests` for the endpoint's OpenAI-compatible API, the `nebius` CLI (subprocess, behind `CliRunner`), vLLM (runs inside the Nebius-managed Endpoint container, not a local dep). `.venv` workflow: `python -m venv .venv && .venv/bin/pip install -e ".[dev]"`.

---

## File Structure

```
pyproject.toml              package + deps + pytest config
.env.example                NEBIUS_* placeholders, no secrets
LICENSE                     MIT
Dockerfile                  Job image (agent + bench)
README.md                   problem · arch · setup · hardware · runtime · cost · proof
src/tuner/
  __init__.py
  config.py                 ALL dataclasses (canonical types)
  pricing.py                cost(metrics, rate) -> CostReport
  bench.py                  aggregate(results,...) + run(client, sweep, ...)
  deploy.py                 deploy(config, runner) / teardown(...) / _parse_url
  agent.py                  propose(history, space, generate) -> Decision
  loop.py                   tune(...) -> Result   (enforces caps, picks winner)
  report.py                 render(result) -> markdown
  adapters.py               real CliRunner / HttpClient / llm_generate
  cli.py                    `tuner tune ...` entrypoint
tests/
  test_pricing.py test_bench.py test_deploy.py
  test_agent.py test_loop.py test_report.py test_cli.py
examples/sample-report.md   committed output from a real run
```

Canonical types live in `config.py` and are imported everywhere — defined once in Task 1, never redefined.

---

### Task 1: Scaffold + canonical types

**Files:**
- Create: `pyproject.toml`, `src/tuner/__init__.py`, `src/tuner/config.py`, `tests/test_config.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "nebius-serverless-tuner"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["requests>=2.31"]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
tuner = "tuner.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
```

- [ ] **Step 2: Write the failing test** — `tests/test_config.py`

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tuner.config'`

- [ ] **Step 4: Write `src/tuner/__init__.py`** (empty) and `src/tuner/config.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Config:
    model: str
    dtype: str = "auto"               # auto | float16 | bfloat16
    max_num_seqs: int = 256
    max_model_len: int = 4096
    quantization: str | None = None   # None | "fp8" | "awq"


@dataclass(frozen=True)
class RequestResult:
    ttft_ms: float
    total_ms: float
    output_tokens: int


@dataclass(frozen=True)
class Metrics:
    ttft_ms_median: float
    tpot_ms_median: float
    throughput_tok_s: float
    total_output_tokens: int
    wall_seconds: float
    concurrency: int
    n_requests: int


@dataclass(frozen=True)
class CostReport:
    gpu_rate_usd_hr: float
    total_cost_usd: float
    cost_per_1m_tok: float


@dataclass(frozen=True)
class Sweep:
    concurrency: int
    input_tokens: int
    output_tokens: int
    n_requests: int


@dataclass(frozen=True)
class SearchSpace:
    dtypes: tuple[str, ...]
    max_num_seqs: tuple[int, ...]
    quantizations: tuple[str | None, ...]


@dataclass(frozen=True)
class Iteration:
    config: Config
    metrics: Metrics
    cost: CostReport


@dataclass(frozen=True)
class NextConfig:
    config: Config
    rationale: str


@dataclass(frozen=True)
class Stop:
    rationale: str


Decision = Union[NextConfig, Stop]


@dataclass(frozen=True)
class Result:
    model: str
    iterations: list[Iteration]
    winning: Iteration
    stopped_reason: str
```

- [ ] **Step 5: Create venv + install, run test to verify it passes**

Run: `python3 -m venv .venv && .venv/bin/pip install -q -e ".[dev]" && .venv/bin/pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: project scaffold + canonical types"
```

---

### Task 2: pricing.cost

**Files:**
- Create: `src/tuner/pricing.py`, `tests/test_pricing.py`

- [ ] **Step 1: Write the failing test** — `tests/test_pricing.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_pricing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tuner.pricing'`

- [ ] **Step 3: Write `src/tuner/pricing.py`**

```python
from __future__ import annotations
from tuner.config import Metrics, CostReport

# GPU hourly rates (USD). VERIFY against live Nebius pricing during the live pass;
# `cost()` itself is rate-agnostic — callers pass the rate for the GPU they deployed on.
NEBIUS_GPU_RATES_USD_HR = {
    "l40s": 1.0,
    "a100": 2.0,
    "h100": 3.0,
}


def cost(metrics: Metrics, gpu_rate_usd_hr: float) -> CostReport:
    total = metrics.wall_seconds / 3600 * gpu_rate_usd_hr
    if metrics.total_output_tokens:
        per_1m = total / (metrics.total_output_tokens / 1_000_000)
    else:
        per_1m = float("inf")
    return CostReport(
        gpu_rate_usd_hr=gpu_rate_usd_hr,
        total_cost_usd=round(total, 6),
        cost_per_1m_tok=round(per_1m, 4) if per_1m != float("inf") else float("inf"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_pricing.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: cost-per-million-tokens pricing"
```

---

### Task 3: bench.aggregate (pure metrics math)

**Files:**
- Create: `src/tuner/bench.py`, `tests/test_bench.py`

- [ ] **Step 1: Write the failing test** — `tests/test_bench.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_bench.py -v`
Expected: FAIL with `ImportError: cannot import name 'aggregate'`

- [ ] **Step 3: Write `aggregate` in `src/tuner/bench.py`**

```python
from __future__ import annotations
import statistics
from typing import Protocol
from tuner.config import RequestResult, Metrics, Sweep


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_bench.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: benchmark metric aggregation"
```

---

### Task 4: bench.run (orchestration behind a Client seam)

**Files:**
- Modify: `src/tuner/bench.py`
- Modify: `tests/test_bench.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_bench.py`)

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_bench.py::test_run_issues_n_requests_and_aggregates -v`
Expected: FAIL with `ImportError: cannot import name 'run'`

- [ ] **Step 3: Add `Client` protocol + `run` to `src/tuner/bench.py`**

```python
import time
from concurrent.futures import ThreadPoolExecutor


class Client(Protocol):
    def chat(self, input_tokens: int, output_tokens: int) -> RequestResult: ...


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_bench.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: benchmark driver behind a Client seam"
```

---

### Task 5: deploy (Nebius CLI behind a CliRunner seam)

**Files:**
- Create: `src/tuner/deploy.py`, `tests/test_deploy.py`

> **Note on CLI flags:** the exact `nebius ai endpoint create` flag surface is pinned against live docs + a real run during the live pass. `deploy.py` isolates every CLI call behind `CliRunner`, so adjusting flags later touches one function. The arg-building and URL parsing are fully tested here.

- [ ] **Step 1: Write the failing test** — `tests/test_deploy.py`

```python
import pytest
from tuner.config import Config
from tuner.deploy import deploy, teardown, DeployError, CliResult


class FakeRunner:
    def __init__(self, result):
        self.result = result
        self.last_args = None

    def run(self, args):
        self.last_args = args
        return self.result


def test_deploy_builds_args_and_parses_url():
    runner = FakeRunner(CliResult(
        stdout='{"status":"running","url":"https://abc.endpoints.nebius.com/v1"}',
        stderr="", returncode=0))
    url = deploy(Config(model="Qwen/Qwen2.5-1.5B-Instruct", quantization="fp8"), runner)
    assert url == "https://abc.endpoints.nebius.com/v1"
    assert "create" in runner.last_args
    assert "Qwen/Qwen2.5-1.5B-Instruct" in runner.last_args
    assert "fp8" in runner.last_args


def test_deploy_raises_on_nonzero():
    runner = FakeRunner(CliResult(stdout="", stderr="boom", returncode=1))
    with pytest.raises(DeployError):
        deploy(Config(model="m"), runner)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_deploy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tuner.deploy'`

- [ ] **Step 3: Write `src/tuner/deploy.py`**

```python
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Protocol
from tuner.config import Config


@dataclass(frozen=True)
class CliResult:
    stdout: str
    stderr: str
    returncode: int


class CliRunner(Protocol):
    def run(self, args: list[str]) -> CliResult: ...


class DeployError(RuntimeError):
    pass


def _endpoint_args(config: Config, name: str) -> list[str]:
    args = [
        "nebius", "ai", "endpoint", "create",
        "--name", name,
        "--model", config.model,
        "--dtype", config.dtype,
        "--max-num-seqs", str(config.max_num_seqs),
        "--max-model-len", str(config.max_model_len),
        "--format", "json",
    ]
    if config.quantization:
        args += ["--quantization", config.quantization]
    return args


def _parse_url(stdout: str) -> str:
    try:
        return json.loads(stdout)["url"]
    except (json.JSONDecodeError, KeyError):
        m = re.search(r"https://\S+", stdout)
        if not m:
            raise DeployError(f"no endpoint URL in CLI output: {stdout!r}")
        return m.group(0)


def deploy(config: Config, runner: CliRunner, name: str = "tuner-endpoint") -> str:
    res = runner.run(_endpoint_args(config, name))
    if res.returncode != 0:
        raise DeployError(res.stderr or "endpoint create failed")
    return _parse_url(res.stdout)


def teardown(name: str, runner: CliRunner) -> None:
    res = runner.run(["nebius", "ai", "endpoint", "delete", "--name", name])
    if res.returncode != 0:
        raise DeployError(res.stderr or "endpoint delete failed")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_deploy.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: endpoint deploy/teardown behind a CliRunner seam"
```

---

### Task 6: agent.propose (LLM behind a Generate seam)

**Files:**
- Create: `src/tuner/agent.py`, `tests/test_agent.py`

- [ ] **Step 1: Write the failing test** — `tests/test_agent.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tuner.agent'`

- [ ] **Step 3: Write `src/tuner/agent.py`**

```python
from __future__ import annotations
import json
import re
from typing import Protocol
from tuner.config import Config, Iteration, SearchSpace, NextConfig, Stop, Decision


class Generate(Protocol):
    def __call__(self, prompt: str) -> str: ...


def _build_prompt(history: list[Iteration], space: SearchSpace) -> str:
    lines = ["You are tuning vLLM serving config to minimise cost-per-1M-tokens.",
             f"Search space: dtypes={space.dtypes}, max_num_seqs={space.max_num_seqs}, "
             f"quantizations={space.quantizations}.", "History (config -> cost/1M, ttft, tpot):"]
    for it in history:
        lines.append(f"- dtype={it.config.dtype} max_num_seqs={it.config.max_num_seqs} "
                     f"quant={it.config.quantization} -> ${it.cost.cost_per_1m_tok}/1M, "
                     f"ttft={it.metrics.ttft_ms_median}ms, tpot={it.metrics.tpot_ms_median}ms")
    lines.append('Reply with JSON: {"action":"continue"|"stop","dtype":...,'
                 '"max_num_seqs":...,"quantization":null|"fp8"|"awq","rationale":"..."}')
    return "\n".join(lines)


def _extract_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise ValueError(f"no JSON in agent reply: {raw!r}")
        return json.loads(m.group(0))


def _in_space(cfg: Config, space: SearchSpace) -> bool:
    return (cfg.dtype in space.dtypes
            and cfg.max_num_seqs in space.max_num_seqs
            and cfg.quantization in space.quantizations)


def propose(history: list[Iteration], space: SearchSpace, generate: Generate) -> Decision:
    data = _extract_json(generate(_build_prompt(history, space)))
    if data.get("action") == "stop":
        return Stop(data.get("rationale", ""))
    last = history[-1].config
    cfg = Config(
        model=last.model,
        dtype=data.get("dtype", last.dtype),
        max_num_seqs=int(data.get("max_num_seqs", last.max_num_seqs)),
        max_model_len=last.max_model_len,
        quantization=data.get("quantization"),
    )
    if not _in_space(cfg, space):
        return Stop(f"proposed config out of search space: {cfg}")
    return NextConfig(cfg, data.get("rationale", ""))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_agent.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: tuning agent behind a Generate seam"
```

---

### Task 7: loop.tune (orchestration + caps + winner)

**Files:**
- Create: `src/tuner/loop.py`, `tests/test_loop.py`

- [ ] **Step 1: Write the failing test** — `tests/test_loop.py`

```python
from tuner.config import (Config, Metrics, SearchSpace, NextConfig, Stop, Sweep)
from tuner.loop import tune

SPACE = SearchSpace(("auto",), (64, 256), (None, "fp8"))
SWEEP = Sweep(concurrency=1, input_tokens=32, output_tokens=10, n_requests=2)


def _metrics(tput):
    return Metrics(50, 10, tput, 1000, 1.0, 1, 2)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tuner.loop'`

- [ ] **Step 3: Write `src/tuner/loop.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_loop.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: tune loop with iteration+budget caps and winner selection"
```

---

### Task 8: report.render

**Files:**
- Create: `src/tuner/report.py`, `tests/test_report.py`

- [ ] **Step 1: Write the failing test** — `tests/test_report.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tuner.report'`

- [ ] **Step 3: Write `src/tuner/report.py`**

```python
from __future__ import annotations
from tuner.config import Result


def render(result: Result) -> str:
    lines = [f"# Tuning report — {result.model}", "",
             "| # | dtype | max_num_seqs | quant | ttft ms | tpot ms | tok/s | cost/1M ($) |",
             "|---|-------|--------------|-------|---------|---------|-------|-------------|"]
    for i, it in enumerate(result.iterations, 1):
        c, m, cr = it.config, it.metrics, it.cost
        lines.append(f"| {i} | {c.dtype} | {c.max_num_seqs} | {c.quantization} | "
                     f"{m.ttft_ms_median} | {m.tpot_ms_median} | {m.throughput_tok_s} | "
                     f"{cr.cost_per_1m_tok} |")
    w = result.winning
    lines += ["",
              f"**Winning config:** dtype={w.config.dtype}, "
              f"max_num_seqs={w.config.max_num_seqs}, quant={w.config.quantization} "
              f"→ ${w.cost.cost_per_1m_tok}/1M tokens "
              f"(ttft {w.metrics.ttft_ms_median} ms, tpot {w.metrics.tpot_ms_median} ms)",
              "",
              f"_Stopped: {result.stopped_reason}_"]
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_report.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: markdown tuning report"
```

---

### Task 9: real adapters + CLI entrypoint

**Files:**
- Create: `src/tuner/adapters.py`, `src/tuner/cli.py`, `tests/test_cli.py`

- [ ] **Step 1: Write the failing test** — `tests/test_cli.py`

```python
from tuner import cli
from tuner.config import Config, Metrics, Stop


def test_cli_run_wires_seams(monkeypatch, capsys):
    # patch the live seams so the CLI runs fully offline
    monkeypatch.setattr(cli, "make_deploy_fn", lambda: (lambda cfg: "https://e/v1"))
    monkeypatch.setattr(cli, "make_bench_fn", lambda url_holder: (lambda url: Metrics(50, 10, 500, 1000, 1.0, 8, 16)))
    monkeypatch.setattr(cli, "make_agent_fn", lambda: (lambda hist, space: Stop("done")))
    rc = cli.main(["tune", "--model", "Qwen/Qwen2.5-1.5B-Instruct",
                   "--max-iters", "1", "--budget-usd", "5", "--gpu-rate", "2.0",
                   "--gpu", "a100"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Winning config" in out
    assert "Qwen/Qwen2.5-1.5B-Instruct" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tuner.cli'`

- [ ] **Step 3: Write `src/tuner/adapters.py`**

```python
from __future__ import annotations
import os
import subprocess
import time
import requests
from tuner.config import RequestResult
from tuner.deploy import CliResult


class SubprocessRunner:
    def run(self, args: list[str]) -> CliResult:
        p = subprocess.run(args, capture_output=True, text=True)
        return CliResult(stdout=p.stdout, stderr=p.stderr, returncode=p.returncode)


class HttpClient:
    """OpenAI-compatible chat client against a deployed endpoint."""
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def chat(self, input_tokens: int, output_tokens: int) -> RequestResult:
        prompt = "word " * input_tokens
        start = time.monotonic()
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": output_tokens, "temperature": 0},
            timeout=120,
        )
        total_ms = (time.monotonic() - start) * 1000
        r.raise_for_status()
        body = r.json()
        out_tok = body.get("usage", {}).get("completion_tokens", output_tokens)
        # vLLM returns no per-token TTFT without streaming; approximate TTFT as
        # total - (out_tok-1)*observed-tpot is circular, so for the non-stream path
        # record ttft == total_ms (documented limitation; streaming variant is future work).
        return RequestResult(ttft_ms=round(total_ms, 2), total_ms=round(total_ms, 2),
                             output_tokens=out_tok)


def llm_generate(prompt: str) -> str:
    """The agent's brain — uses Nebius Token Factory (cheap) by default."""
    base = os.environ.get("AGENT_LLM_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
    key = os.environ["NEBIUS_API_KEY"]
    model = os.environ.get("AGENT_LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
    r = requests.post(f"{base.rstrip('/')}/chat/completions",
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": model, "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 256, "temperature": 0}, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
```

- [ ] **Step 4: Write `src/tuner/cli.py`**

```python
from __future__ import annotations
import argparse
import os
from tuner.config import Config, SearchSpace, Sweep, Iteration, SearchSpace as _SS
from tuner.loop import tune
from tuner.report import render
from tuner.bench import run as bench_run
from tuner.agent import propose
from tuner.deploy import deploy, teardown
from tuner import adapters

DEFAULT_SPACE = SearchSpace(dtypes=("auto", "bfloat16"),
                            max_num_seqs=(64, 256),
                            quantizations=(None, "fp8"))


def make_deploy_fn():
    runner = adapters.SubprocessRunner()
    return lambda cfg: deploy(cfg, runner)


def make_bench_fn(sweep: Sweep):
    def bench_fn(url: str):
        client = adapters.HttpClient(url, os.environ["NEBIUS_API_KEY"],
                                     model=os.environ.get("ENDPOINT_MODEL", ""))
        return bench_run(client, sweep)
    return bench_fn


def make_agent_fn():
    return lambda hist, space: propose(hist, space, adapters.llm_generate)


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
    result = tune(model=args.model, base_config=Config(model=args.model),
                  search_space=DEFAULT_SPACE, sweep=sweep, gpu_rate=args.gpu_rate,
                  deploy_fn=make_deploy_fn(), bench_fn=make_bench_fn(sweep),
                  agent_fn=make_agent_fn(), max_iters=args.max_iters,
                  budget_usd=args.budget_usd)
    print(render(result))
    return 0
```

> The test monkeypatches `make_deploy_fn`/`make_bench_fn`/`make_agent_fn`, so `main` must call them by name (it does). For the test's `make_bench_fn` signature, accept any single arg: change the test-facing call to `make_bench_fn(sweep)` — already the case.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: PASS (1 passed)

> If the monkeypatch signature mismatches (`make_bench_fn` takes `sweep`, test passes `url_holder`), align the test to `monkeypatch.setattr(cli, "make_bench_fn", lambda sweep: (lambda url: Metrics(...)))`. Keep the production signature `make_bench_fn(sweep)`.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS (all tests, ~13)

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: real adapters (CLI runner, HTTP client, agent LLM) + tuner CLI"
```

---

### Task 10: Packaging — Dockerfile, README, LICENSE, .env.example, sample report

**Files:**
- Create: `Dockerfile`, `README.md`, `LICENSE`, `.env.example`, `examples/sample-report.md`

- [ ] **Step 1: Write `LICENSE`** (MIT, author "Artur Arapov", year 2026) — standard MIT text.

- [ ] **Step 2: Write `.env.example`**

```bash
# Copy to .env and fill — NEVER commit .env
NEBIUS_API_KEY=
ENDPOINT_MODEL=Qwen/Qwen2.5-1.5B-Instruct
AGENT_LLM_BASE_URL=https://api.tokenfactory.nebius.com/v1
AGENT_LLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
```

- [ ] **Step 3: Write `Dockerfile`** (the Job image)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .
ENTRYPOINT ["tuner"]
```

- [ ] **Step 4: Write `README.md`** — sections: what it is; architecture diagram (copy from spec); **Setup** (`python -m venv .venv && .venv/bin/pip install -e ".[dev]"`); **Hardware** (endpoint GPU type, e.g. 1× L40S/A100; Job runs CPU-only); **Run** (`tuner tune --model ... --gpu a100 --gpu-rate <rate> --max-iters 3 --budget-usd 5`); **Expected output** (link `examples/sample-report.md`); **Runtime + cost** (≈ N min, ≈ $X on $100 promo); **Proof** (placeholders for endpoint URL + job-log screenshots filled during the live pass); **How it works on Nebius** (Endpoint = vLLM serving; Job = agent+bench); **Tests** (`.venv/bin/pytest -q`, fully offline); **License**.

- [ ] **Step 5: Write `examples/sample-report.md`** — paste the `render()` output format with representative numbers, labelled "sample (schema); real numbers captured during the live run."

- [ ] **Step 6: Run full suite once more**

Run: `.venv/bin/pytest -q`
Expected: PASS (all green)

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "docs: Dockerfile, README, LICENSE, env example, sample report"
```

---

## Post-implementation phase (live + submission — Claude-driven, Artur authorized)

These are NOT TDD coding tasks; they run after the suite is green, on Artur's Nebius account + $100 credits, bounded by `--max-iters` and `--budget-usd`. Tracked separately so the code is committable independently.

- [ ] **L1 — Pin the live CLI surface:** verify `nebius` CLI auth + the exact `endpoint create`/`delete` flags and the GPU-rate table against live docs; adjust `deploy.py` / `pricing.NEBIUS_GPU_RATES_USD_HR` if they differ; re-run suite.
- [ ] **L2 — Deploy + bounded tune pass:** push repo to `github.com/art87able/nebius-serverless-tuner`; run `tuner tune` for a small model (Qwen2.5-1.5B-Instruct) with `--max-iters 3 --budget-usd 5`; `stop`/delete the endpoint after. Capture proof: endpoint URL, job logs, screenshots, the real `examples/sample-report.md`.
- [ ] **L3 — Blog + submission:** write ≥600-word DEV Community post tagged `#NebiusServerlessChallenge` (problem → agentic approach → Jobs+Endpoints architecture → results + proof screenshots, original narrative ≠ README); export PDF; ≤100-word summary; fill the AI Academy submit form (repo URL · blog URL · PDF · summary · optional video). Confirm with Artur before publishing/submitting.

---

## Self-Review

**Spec coverage:** deploy ✓(T5) · bench ✓(T3,T4) · pricing ✓(T2) · agent ✓(T6) · loop ✓(T7) · report ✓(T8) · floor (deploy+bench+pricing+report single config) ✓ · ceiling (agent+loop bounded caps) ✓(T7 budget/iter tests) · repo layout ✓(T1,T10) · Dockerfile/README/LICENSE/.env ✓(T10) · mocked-tests-no-billing ✓(every task uses fakes) · deliverables/blog/form ✓(post-impl L3) · risks (caps, best-so-far winner, CLI isolation) ✓.

**Placeholder scan:** no "TODO/TBD" in code steps; all code blocks complete. The two deferred-to-live items (exact CLI flags, GPU rates) are concrete best-guesses with tests against the interface, explicitly pinned in L1 — not placeholders in the code.

**Type consistency:** `Config(model,dtype,max_num_seqs,max_model_len,quantization)`, `Metrics(ttft_ms_median,tpot_ms_median,throughput_tok_s,total_output_tokens,wall_seconds,concurrency,n_requests)`, `CostReport(gpu_rate_usd_hr,total_cost_usd,cost_per_1m_tok)`, `Iteration(config,metrics,cost)`, `Decision=NextConfig|Stop`, `Result(model,iterations,winning,stopped_reason)` — used identically across T1–T10. `deploy_fn/bench_fn/agent_fn` signatures match between `loop.tune` (T7) and `cli` (T9).
