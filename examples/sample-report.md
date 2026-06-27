> **Real live run** — captured 2026-06-25 on a Nebius L40S endpoint (`gpu-l40s-a`, `1gpu-8vcpu-32gb`)
> serving `Qwen/Qwen2.5-1.5B-Instruct`. Bounded `--max-iters 3 --budget-usd 5`; the agent stopped at
> max-iters. Endpoint torn down in `finally` — `nebius ai endpoint list` empty afterwards (~$0.50 total).
> Raw log: [`proof/live-run-2026-06-25-run2.log`](../proof/live-run-2026-06-25-run2.log).

# Tuning report — Qwen/Qwen2.5-1.5B-Instruct

| # | dtype | max_num_seqs | quant | ttft ms | tpot ms | tok/s | cost/1M ($) |
|---|-------|--------------|-------|---------|---------|-------|-------------|
| 1 | auto | 256 | None | 793.38 | 0.0 | 341.34 | 1.2612 |
| 2 | bfloat16 | 256 | None | 705.55 | 0.0 | 419.96 | 1.0253 |

**Winning config:** dtype=bfloat16, max_num_seqs=256, quant=None → $1.0253/1M tokens (ttft 705.55 ms, tpot 0.0 ms)

_Stopped: max_iters reached_

---

**Honest caveats on these numbers**

- **The agent cut cost ~19%** (`$1.2612 → $1.0253` per 1M output tokens) by switching `dtype: auto → bfloat16`,
  which raised throughput `341 → 420 tok/s` on the same GPU. That delta — found and measured automatically —
  is the result that matters.
- **`tpot` reads `0.0` and `ttft` is really *total request latency*.** The benchmark uses the non-streaming
  chat-completions path, which returns one response after the full generation, so time-to-first-token can't be
  separated from per-output-token time (`adapters.py`). The `ttft` column is therefore the whole-request median,
  and `tpot` is unmeasured (recorded as 0). Cost/1M and tok/s are derived from total wall-time ÷ output tokens
  and are unaffected. A streaming benchmark variant (real TTFT + TPOT) is the obvious next improvement.
- **The fp8 row from the earlier schema sample didn't run** — this bounded pass stopped at `--max-iters` after
  two iterations rather than exhausting the search space.
