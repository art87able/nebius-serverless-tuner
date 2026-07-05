# Serverless Tuner

**An agent that does AI performance engineering for you on Nebius Serverless.** Point it at any
open model; it deploys the model as a Nebius Serverless **Endpoint** (vLLM), benchmarks it from a
Serverless **Job**, then an LLM **agent** reads the results, proposes a better serving config,
redeploys, re-benchmarks — and hands you a **cost-per-token report, the winning config,
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

### Run it as a Nebius Serverless Job

The same loop runs fully inside a **Serverless AI Job** — the Job drives the agent + benchmark,
deploys/tears down the Endpoint from within the container ([`job-entrypoint.sh`](job-entrypoint.sh)
authenticates the in-container `nebius` CLI from a short-lived IAM token):

```bash
docker buildx build --platform linux/amd64 -t $REGISTRY/tuner:job-v2 . && docker push $REGISTRY/tuner:job-v2

# store the (short-lived) IAM token in MysteryBox — never in the job spec:
nebius mysterybox secret create --parent-id $PROJECT --name tuner-job-creds \
  --secret-version-payload "[{\"key\":\"NEBIUS_IAM_TOKEN\",\"string_value\":\"$(nebius iam get-access-token)\"}]"

nebius ai job create --parent-id $PROJECT --name tuner-job \
  --image $REGISTRY/tuner:job-v2 \
  --args "tune --model Qwen/Qwen2.5-1.5B-Instruct --gpu l40s --gpu-rate 2.00 --max-iters 2 --budget-usd 5 --concurrency 8 --requests 16" \
  --platform cpu-d3 --preset 4vcpu-16gb --subnet-id $SUBNET_ID --timeout 2h \
  --env-secret NEBIUS_IAM_TOKEN=tuner-job-creds \
  --env NEBIUS_PARENT_ID=$PROJECT --env ENDPOINT_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
  --env AGENT_LLM_BASE_URL=endpoint

nebius ai job logs $(nebius ai job get-by-name --name tuner-job --format jsonpath='{.metadata.id}') --follow
```

The Job is CPU-only (`cpu-d3`); the GPU lives solely in the Endpoint it manages.
`AGENT_LLM_BASE_URL=endpoint` makes the run **self-contained**: the agent's brain is the very
endpoint the tuner deploys (late-bound to the current iteration's URL), so no external LLM API
key is needed — the only credential the Job carries is a short-lived IAM token.

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
- **Cost:** dominated by Endpoint GPU-hours during benchmarking; `--budget-usd` puts a hard cap
  on it. GPU rates vary by platform — `deploy.py` isolates every CLI call, so adjusting rates or
  `endpoint create` flags touches one function.

## Proof it runs on Nebius

**Live run — 2026-06-25, Nebius L40S (`gpu-l40s-a`), `Qwen/Qwen2.5-1.5B-Instruct`:**

- The agent tuned `dtype: auto → bfloat16` and cut cost **$1.2612 → $1.0253 per 1M tokens** (≈19%),
  raising throughput `341 → 420 tok/s` on the same GPU.
- Bounded (`--max-iters 2 --budget-usd 5`), ran end-to-end in ~18 min, **torn down in `finally`** —
  `nebius ai endpoint list` empty afterwards.
- Raw log: [`proof/live-run-2026-06-25-run2.log`](proof/live-run-2026-06-25-run2.log).
- Real report (with the measurement caveats on the latency columns): [`examples/sample-report.md`](examples/sample-report.md).

**Live run as a Serverless Job — 2026-07-05, self-contained (endpoint-as-agent-brain):**

- The full loop ran **inside a Serverless AI Job** (`cpu-d3`, CPU-only): the container authenticated
  the in-baked `nebius` CLI from a MysteryBox-injected IAM token, deployed the L40S endpoint,
  benchmarked ~**1,016 tok/s** at **$0.5466 / 1M tokens**, and consulted the agent served by
  **the deployed endpoint itself** (`AGENT_LLM_BASE_URL=endpoint`) — no external LLM API.
- Job finished `COMPLETED`; `nebius ai endpoint list` empty afterwards (teardown in `finally`).
- Raw log + job metadata: [`proof/job-run-2026-07-05-run2.log`](proof/job-run-2026-07-05-run2.log).

## Tests

```bash
.venv/bin/pytest -q
```

The full suite runs **offline** — no network, no GPU, no billing — by faking the `nebius` CLI,
the endpoint client, and the agent's LLM.

## License

[MIT](LICENSE).
