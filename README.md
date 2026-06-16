# Serverless Tuner

**An agent that does AI performance engineering for you on Nebius Serverless.** Point it at any
open model; it deploys the model as a Nebius Serverless **Endpoint** (vLLM), benchmarks it from a
Serverless **Job**, then an LLM **agent** reads the results, proposes a better serving config,
redeploys, re-benchmarks — and hands you an honest **cost-per-token report, the winning config,
and the reasoning trace.**

> Entry for the **Nebius Serverless AI Builders Challenge** (#NebiusServerlessChallenge).
> Domain: AI & ML. The benchmark is the agent's sensor; the agent is what turns a one-off
> benchmark into an autonomous optimizer.

## Why it exists

"What will it cost and how fast will it be to serve model *M*?" is the first question before any
deployment — and answering it means manually deploying, load-testing, tweaking vLLM flags, and
repeating. Serverless Tuner automates that loop and produces a reproducible report a teammate can
rerun.

## How it works on Nebius

```
Serverless JOB  ── runs the agent loop ──────────────────────────────────┐
   │   ┌─ benchmark probe: concurrency × seq-len against the Endpoint      │
   │   │      → TTFT · TPOT · throughput (tok/s) · $/1M-tok                 │
   │   ▼                                                                    │
   │   AGENT: read results → propose config (dtype, max_num_seqs, quant) ─┐ │
   ▼                                                                      ▼ │
Serverless ENDPOINT (vLLM, OpenAI-compatible /v1/chat/completions) ◄── redeploy
   │                                                                        │
   └────► converged OR iteration/budget cap ──► REPORT (table + trace + winner)
```

- **Endpoint** = the model-under-test, served by a Nebius-managed vLLM container and re-served as
  the agent changes the config.
- **Job** = the tuner itself (agent + benchmark), run to completion.

Both core Serverless products, each used for what it is for.

## Architecture

The whole loop sits behind three injected seams so it is fully unit-testable offline:

| Unit | Responsibility | Seam |
|---|---|---|
| `deploy.py` | create / redeploy / teardown an Endpoint for a config | `CliRunner` (the `nebius` CLI) |
| `bench.py` | drive load, measure latency/throughput | `Client` (OpenAI-compatible HTTP) |
| `pricing.py` | GPU-seconds + tokens → `$/1M-tok` | — (pure) |
| `agent.py` | read history → propose next config or stop | `Generate` (an LLM) |
| `loop.py` | orchestrate deploy→bench→agent, enforce caps, pick winner | — |
| `report.py` | render the table + trace + winning config | — |

Every iteration's endpoint is **torn down in a `finally` block** so a run can never leave a
billing endpoint behind.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env        # then fill in NEBIUS_API_KEY etc.
```

Requires the [`nebius` CLI](https://docs.nebius.com/cli) authenticated to your tenant, plus a
Nebius API key.

## Run

```bash
tuner tune \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --gpu a100 --gpu-rate 2.00 \
  --max-iters 3 --budget-usd 5 \
  --concurrency 8 --requests 16
```

- `--gpu-rate` is the USD/hour for the GPU you deploy on (from Nebius pricing).
- `--max-iters` and `--budget-usd` are **hard caps** — the loop stops at whichever it hits first.

### Expected output

A markdown report like [`examples/sample-report.md`](examples/sample-report.md): a per-iteration
table (dtype · max_num_seqs · quant · TTFT · TPOT · tok/s · $/1M), the winning config, and why the
agent stopped.

## Hardware, runtime & cost

- **Endpoint:** 1 GPU (e.g. L40S / A100), scale-to-zero when stopped. The model-under-test is
  small (≈1.5B) so a single GPU is plenty.
- **Job (tuner):** CPU-only — it just issues HTTP requests and a few agent calls.
- **Runtime:** ≈ a few minutes per iteration (deploy + warm-up + benchmark); a 3-iteration run is
  well under an hour.
- **Cost:** dominated by Endpoint GPU-hours during benchmarking. A bounded 3-iteration run on a
  small model stays inside the challenge's **$100 promo credits**; `--budget-usd` enforces it.

> Exact GPU rates and the precise `nebius ai endpoint create` flags are pinned against live docs +
> a real run; `deploy.py` isolates every CLI call so adjusting them touches one function.

## Proof it runs on Nebius

_Filled during the live pass:_

- Endpoint URL: _<pending>_
- Job logs: _see `proof/`_
- Screenshots: _see `proof/`_
- Real report: `examples/sample-report.md` (replaces the schema sample)

## Tests

```bash
.venv/bin/pytest -q
```

The full suite runs **offline** — no network, no GPU, no billing — by faking the `nebius` CLI,
the endpoint client, and the agent's LLM.

## License

[MIT](LICENSE).
