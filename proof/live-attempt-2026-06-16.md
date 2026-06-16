# Live attempt — 2026-06-16

First real deploy on Nebius (tenant `qdbeb`, project eu-north1 `project-e00w6e05pr004483zzrpzc`).

- **CREATE accepted** — endpoint `aiendpoint-e00yabep2hk9emy07b`, public endpoint `195.242.28.162:8000`, image `vllm/vllm-openai:v0.18.0-cu130`, platform `gpu-l40s-a`, preset `1gpu-8vcpu-32gb`. The deploy recipe is correct (Nebius accepted every flag).
- **BLOCKED on GPU provisioning** — instance stayed `STOPPED` / endpoint `PROVISIONING` for ~18 min, never started. Classic zero-GPU-quota on a fresh account (promo = credits, not quota).
- **Torn down cleanly** — `endpoint list` returned `{}` (no lingering resource, no continued billing).

**Verified output shape** (drove the `_parse_url` fix): the OpenAI base URL = `http://` + `status.public_endpoints[0]` + `/v1`.

**Next:** request GPU quota for the correct project in the Nebius console, then re-run `tuner tune`.
