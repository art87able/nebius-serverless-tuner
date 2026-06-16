# Live runbook — the bounded Nebius pass

The offline build is complete and tested. This is the live phase: one bounded tune pass on a real
Nebius account + $100 promo credits, to capture proof. **Hard-bounded** by `--max-iters` and
`--budget-usd`; the endpoint is always torn down in a `finally`.

## Prereqs (confirmed 2026-06-16)

- `nebius` CLI v0.12.x installed + authenticated (profile `Artur [default]`).
- A project/parent is configured in the CLI.
- `$NEBIUS_API_KEY` (Token Factory) for the agent's brain.

## Verified recipe (Nebius deploy-model tutorial)

```bash
AUTH_TOKEN=$(openssl rand -hex 32)
SUBNET_ID=$(nebius vpc subnet list --format jsonpath='{.items[0].metadata.id}')

nebius ai endpoint create \
  --name tuner-endpoint \
  --image vllm/vllm-openai:v0.18.0-cu130 \
  --container-command "python3 -m vllm.entrypoints.openai.api_server" \
  --args "--model Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8000 --dtype bfloat16 --max-num-seqs 256 --max-model-len 4096" \
  --platform gpu-l40s-a --preset 1gpu-8vcpu-32gb \
  --public --container-port 8000 \
  --auth token --token "$AUTH_TOKEN" \
  --shm-size 16Gi --subnet-id "$SUBNET_ID" \
  --format json
```
`deploy.py` builds exactly this; `cli.py` generates `AUTH_TOKEN` per run and reuses it for the
benchmark client; `adapters.get_subnet_id` resolves `SUBNET_ID`.

Call (OpenAI-compatible): `POST http(s)://<endpoint>/v1/chat/completions`, `Authorization: Bearer $AUTH_TOKEN`.
Delete: `nebius ai endpoint delete --id <id>` (teardown resolves name→id first).

## Live-only unknowns to confirm on the first deploy (then pin in code)

1. **Create output shape** — exact JSON path to the endpoint URL/IP and the id. `deploy._parse_url`
   / `_parse_id` try several paths + a regex fallback; confirm and tighten against a real response.
   (Tutorial reaches the endpoint by public IP → URL may be `http://<ip>`.)
2. **`--args` quoting** — confirm vLLM receives the space-joined arg string as one token correctly
   via the CLI; adjust quoting if the CLI splits it.
3. **GPU rate** — read the real L40S `gpu-l40s-a` USD/hr from Nebius pricing → pass as `--gpu-rate`
   and update `pricing.NEBIUS_GPU_RATES_USD_HR`.
4. **Cold start vs health** — vLLM downloads the model at startup (~minutes); add a health poll
   before benchmarking if needed.

## The bounded run

```bash
tuner tune --model Qwen/Qwen2.5-1.5B-Instruct \
  --gpu l40s --gpu-rate <real-rate> --max-iters 3 --budget-usd 5 \
  --concurrency 8 --requests 16
```

Capture into `proof/`: the create response, `nebius ai endpoint list`, vLLM/job logs, a screenshot,
and the rendered report → replace `examples/sample-report.md` with the real one.

## After the run

- Confirm teardown: `nebius ai endpoint list` shows no `tuner-endpoint`.
- Blog (≥600 words, `#NebiusServerlessChallenge`) + ≤100-word summary + submit via the AI Academy form.
