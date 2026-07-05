# I built an agent that makes LLM serving cheaper — on Nebius Serverless

**#NebiusServerlessChallenge**

## The 100-word version

Serving an open LLM has a dozen knobs — dtype, quantization, batch sizing — and the wrong combo quietly doubles your cost per token. **Nebius Serverless Tuner** automates the search: a Nebius Serverless **Job** runs the whole loop — deploy the model as a Serverless **Endpoint** (vLLM), benchmark it under load, and let an LLM agent (served by that same endpoint) propose new serving configs to minimize **$/1M tokens** — then writes a cost report and tears everything down. On a live L40S run it tuned `auto → bfloat16` and cut cost from **$1.26 → $1.03 per 1M tokens** automatically. Built test-first; the endpoint is always released in a `finally`.

## The problem

LLM inference cost isn't one number you look up — it's an optimization. The same model on the same GPU can cost wildly different amounts per token depending on how you serve it: the compute dtype, whether you quantize, how many sequences you batch concurrently, the max context length. Picking these by hand means standing up an endpoint, hammering it with a load test, reading the throughput, changing one flag, and repeating. It's exactly the kind of tedious, measurable loop that should be automated — and exactly what serverless infrastructure is good at, because you only pay while the GPU is actually up.

## What I built

**Nebius Serverless Tuner** is a small Python tool with one job: find the cheapest way to serve a given open model, with evidence. It uses the two core Serverless primitives directly:

- **Endpoint** — it deploys the model as a vLLM container on a GPU platform (e.g. `gpu-l40s-a`), exposed as an OpenAI-compatible API.
- **Job** — the tuner itself runs as a CPU-only Serverless Job: it drives a fixed load sweep (concurrency, request count, fixed input/output token lengths) against the endpoint, records latency + throughput, and manages the endpoint's lifecycle from inside the container.

Around that sits a tuning loop:

1. **Deploy** the current serving config as an Endpoint, then **wait for readiness** (poll `/v1/models` until vLLM has finished loading — endpoints return their address before the model is live).
2. **Benchmark** it and compute **cost per 1M tokens** = GPU-$/hr × wall-seconds ÷ output-tokens.
3. An **LLM agent** looks at the history so far and proposes the next config to try from a small search space (dtype, max-num-seqs, fp8 quantization), or stops. The neat part: with `AGENT_LLM_BASE_URL=endpoint` the agent's brain is **the deployed endpoint itself** — the model under test does its own tuning, so the run needs no external LLM API at all.
4. **Redeploy** the new config (tearing the previous endpoint down first), repeat — bounded by `--max-iters` and `--budget-usd`.
5. **Report**: the winning config, a per-iteration table, and the measured cost numbers — then tear the endpoint down in a `finally` so nothing keeps billing.

## The live result

On a real Nebius L40S endpoint serving `Qwen/Qwen2.5-1.5B-Instruct`, a bounded two-iteration run:

| # | dtype | throughput (tok/s) | cost / 1M tokens |
|---|-------|--------------------|------------------|
| 1 | auto | 341 | $1.26 |
| 2 | **bfloat16** | **420** | **$1.03** |

The agent tried the default (`auto`) first, then proposed `bfloat16`, measured a ~19% cost reduction, and reported it as the winner — **$1.0253 / 1M tokens**. The whole run deployed, benchmarked, tuned, and tore down in ~18 minutes, ending with an empty endpoint list (zero lingering spend).

And the full loop also ran **as an actual Serverless Job** (2026-07-05): a CPU-only `cpu-d3` Job container authenticated the `nebius` CLI from a short-lived IAM token (stored in MysteryBox, injected via `--env-secret` — never in the job spec), deployed the L40S endpoint, benchmarked it at ~1,000 tok/s ($0.55/1M), consulted the endpoint-hosted agent, and tore everything down — job state `COMPLETED`, endpoint list empty. Same code, zero laptop involvement.

## Practical notes

- **Endpoints return before they're ready.** The create call hands back the address while the GPU is still provisioning and vLLM is still downloading the model (~15 min cold start). Benchmarking immediately fails; poll the API until it answers first.
- **The address shape isn't fixed.** Sometimes the reachable address comes back as a bare `IP:PORT`, sometimes as an `https://…tunnel…nebius.cloud` URL. The tuner appends `/v1` to whichever shape comes back (covered by a regression test).
- **Bounded cost experiments are cheap — with one condition.** You only pay while the endpoint is up, but release it in a `finally`: a crash otherwise leaves a GPU billing.
- **The model under test can be its own tuning brain.** Routing the agent's calls to the endpoint the tuner just deployed removes the only external dependency: one late-bound URL, and the whole system runs self-contained on Serverless.

## Try it

The tool is open source, test-first (24 unit tests that run fully offline), and bounded by design. Point it at a model and a GPU rate, and it'll tell you the cheapest way to serve it — with the benchmark to prove it.

`github.com/art87able/nebius-serverless-tuner`

*Built for the Nebius Serverless AI Builders Challenge.* **#NebiusServerlessChallenge**
