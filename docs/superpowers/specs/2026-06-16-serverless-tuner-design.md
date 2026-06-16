# Serverless Tuner — design spec

*Nebius Serverless AI Builders Challenge entry · 2026-06-16 · status: approved (brainstorm), pending implementation plan*

## Gist

An **agentic AI-performance-engineering assistant for Nebius Serverless.** Point it at any open
model; it deploys the model as a Nebius Serverless **Endpoint** (vLLM), runs a benchmarking
**Job**, then an **agent** reads the results, hypothesizes a better serving config, redeploys,
re-benchmarks, and converges on an honest **cost/perf report + the winning config + its reasoning
trace.** The benchmark is the agent's sensor; the agent is what makes it a reference example
rather than a script.

- **Challenge:** Nebius Serverless AI Builders Challenge — deadline **2026-06-30 23:59 UTC**, winners 2026-07-09.
- **Domain (form):** AI & ML. **Special-award target:** Agentic AI Workflows.
- **Why it wins:** deep dual-product usage (Jobs + Endpoints), high originality + educational
  value (a rigorous, reproducible cost/perf optimizer few entrants will match), plays to the
  author's AI-performance-engineering strength, and doubles as teaching material.

## Constraints & facts (verified)

- The challenge requires ≥1 **Serverless Job or Endpoint** run during the window — **Token Factory
  inference does NOT count** (verified against T&C + AI Academy "Getting Started", 2026-06-15/16).
- A valid submission = **public repo** (code using Jobs/Endpoints · Dockerfile/public image ·
  README with setup/hardware/expected-output/runtime+cost · OSS license · no secrets) **+ ≥600-word
  blog post** tagged `#NebiusServerlessChallenge` (links repo, explains problem/architecture/
  implementation/results, with screenshots/logs proving it ran on Nebius) **+ optional 3–10 min
  video**. Submitted via the AI Academy form (repo URL · blog URL · blog PDF · ≤100-word summary ·
  optional video link). One entry per Tenant ID.
- **$100 promo credits** applied. **Nebius has no hard spend cap** → every live loop is
  hard-bounded by iteration count *and* a credit budget; endpoints are `stop`ped when idle.
- Mechanism (Nebius docs, 2026-06-15): Endpoints deploy via `nebius` CLI as a **vLLM container
  serving any HF model**, exposing **OpenAI-compatible `/v1/chat/completions`** with token auth
  (~5 min spin-up; `stop` pauses billing). Jobs run a Dockerfile to completion (training, batch,
  benchmarking). *Exact CLI invocations are pinned during implementation against the live docs +
  a real run.*

## Architecture

```
Serverless JOB  ── runs the agent loop ──────────────────────────────────┐
   │   ┌─ benchmark probe: sweep concurrency × seq-len against the Endpoint │
   │   │      → TTFT · TPOT · throughput (tok/s) · $/1M-tok                  │
   │   ▼                                                                    │
   │   AGENT: read results → hypothesize config (max-num-seqs, dtype/quant, │
   │      max-model-len, batching, spec-decode) → apply ──┐                 │
   ▼                                                       ▼                │
Serverless ENDPOINT (vLLM, OpenAI-compatible) ◄── redeploy with new config  │
   │                                                                        │
   └────► converged OR iteration/budget cap ──► REPORT (curves + trace + winning config)
```

Two products, used for what each is for: **Endpoint** = the model-under-test being served and
re-served across configs; **Job** = the agent + benchmark runs to completion.

### Components (each independently testable)

| Unit | Purpose | Interface | Depends on |
|---|---|---|---|
| `deploy/` | create / redeploy / teardown a Serverless Endpoint for a given config | `deploy(config) -> endpoint_url`; `teardown()` | Nebius CLI (mocked in tests) |
| `bench/` | drive load at an endpoint, measure latency/throughput/cost | `run(endpoint_url, sweep) -> Metrics` | an OpenAI-compatible URL |
| `pricing/` | turn raw GPU-seconds + tokens into `$/1M-tok` | `cost(metrics, gpu_rate) -> CostReport` | Nebius price table (constant) |
| `agent/` | read Metrics → propose next config → decide stop | `step(history) -> (Config \| STOP, rationale)` | an LLM (`generate()` seam) |
| `loop/` | orchestrate deploy→bench→agent, enforce caps | `tune(model, budget, max_iters) -> Result` | all of the above |
| `report/` | render curves + decision trace + winning config | `render(Result) -> markdown/html` | — |

The `generate()` seam (an env-selected LLM call) lets the agent's brain run on any backend and be
**fully mocked** in tests — the entire loop runs on canned strings/metrics with zero network and
zero GPU in CI.

## Floor + ceiling (ambition control)

- **Floor — always a valid, strong entry:** `deploy` + `bench` + `pricing` + `report` for a
  *single* config = an honest cost/perf report for serving model M on Nebius Serverless. Submittable
  on its own.
- **Ceiling — the differentiator:** the `agent` + `loop` autonomous tune, **hard-bounded** to
  `max_iters` and a `budget_usd`. Model-under-test is small (e.g. Qwen2.5-1.5B-Instruct) so live
  GPU time stays cheap (well inside $100). The agent's brain can run on a cheap/cheap-local model.

## Repo layout — `art87able/nebius-serverless-tuner`

```
README.md          problem · architecture · setup · hardware · expected output · runtime + cost · proof
LICENSE            MIT
Dockerfile         the Job image (agent + bench); Endpoint uses a public vLLM image
pyproject.toml     deps; .venv workflow
.env.example       NEBIUS_* placeholders — NO secrets
src/tuner/
  deploy.py  bench.py  pricing.py  agent.py  loop.py  report.py  config.py
tests/             unit tests; fake Nebius CLI + canned metrics + mocked agent
examples/          sample report (committed) from a real run
docs/
  superpowers/specs/   this spec
  field-notes.md       build narrative (feeds the blog)
```

No committed secrets/keys/tokens; `.env.example` only; results/artifacts gitignored except the
one committed sample report.

## Deliverables → AI Academy submit form

- **Repo URL** ✅ (above), public, MIT.
- **Domain:** AI & ML.
- **Blog post** ≥600 words on **DEV Community**, tagged `#NebiusServerlessChallenge` — original
  narrative: the problem (serving cost/perf is opaque; tuning is manual toil), the agentic
  approach, the Nebius Jobs+Endpoints architecture, implementation, and the real results +
  screenshots/logs proving the Nebius run. Drafted here; published + PDF-exported.
- **Short summary** ≤100 words.
- **Video** (optional, stretch) 3–10 min walkthrough.

## Division of labor

- **Claude builds everything offline:** all code, scripts, Dockerfiles, the agent, README, blog
  draft, summary — with **mocked tests and zero live billing** (fake Nebius CLI, canned benchmark
  numbers, mocked agent brain).
- **Live + publish handled by Claude** (Artur authorized, 2026-06-16) via the logged-in browser /
  CLI on Artur's account + $100 credits: run the bounded live tune pass, capture proof
  (endpoint URL, job logs, screenshots), publish the blog, fill the form. **Hard rule:** every
  live run is bounded by `max_iters` + `budget_usd`, endpoints `stop`ped when idle, and any spend
  beyond a small pre-agreed cap is confirmed with Artur first (no spend cap on the account).

## Rubric fit (the 6 judged criteria)

1. **Technical implementation** — clean vLLM endpoint + job, working end-to-end, real metrics.
2. **Reproducibility** — Dockerfile + README (hardware/runtime/cost) + deploy script + committed sample report.
3. **Educational content** — blog teaches serverless serving cost/perf + agentic tuning.
4. **Product-usage depth** — BOTH Jobs and Endpoints, each used idiomatically.
5. **Real-world usefulness** — answers a question every practitioner has ("what'll it cost + how fast?") and automates the tuning toil.
6. **Originality** — an *agent* that does performance engineering on serverless; few will submit this.

## Testing

TDD. Every unit has tests against a **fake Nebius CLI + canned endpoint responses + mocked agent
brain**; the `loop` is tested on a replayed metrics trace. The eval/report renders deterministically
from canned `Result`s. **No live billing in CI.** The single bounded live run is the manual
verification step, captured as proof artifacts.

## Risks & mitigations

- *No spend cap* → hard `max_iters` + `budget_usd` caps in `loop`; `stop` endpoints when idle; small model-under-test.
- *Agent makes a bad/looping config choice* → bounded iterations, a config validator, and a "best-so-far" fallback so the report is always sane.
- *Endpoint cold-start / deploy flakiness* → deploy with retry + health-check before benchmarking; `report` records partials.
- *Exact CLI surface differs from yesterday's notes* → pin against live docs + a real run during implementation; `deploy.py` isolates all CLI calls behind one interface.

## Out of scope (YAGNI)

- No Unstuck coupling. No multi-model leaderboard (one model-under-test at a time). No web UI
  beyond the rendered report. No distributed/multi-GPU tuning. Agentic special-award framing is a
  lens on the same artifact, not extra subsystems.
