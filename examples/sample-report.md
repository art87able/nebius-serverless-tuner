> **Sample output** — illustrates the report schema. Real numbers are captured during the live tune pass and replace this file.

# Tuning report — Qwen/Qwen2.5-1.5B-Instruct

| # | dtype | max_num_seqs | quant | ttft ms | tpot ms | tok/s | cost/1M ($) |
|---|-------|--------------|-------|---------|---------|-------|-------------|
| 1 | auto | 256 | None | 182.4 | 11.9 | 1840.0 | 0.226 |
| 2 | bfloat16 | 256 | None | 171.2 | 10.4 | 2105.0 | 0.198 |
| 3 | bfloat16 | 256 | fp8 | 168.9 | 8.7 | 2630.0 | 0.158 |

**Winning config:** dtype=bfloat16, max_num_seqs=256, quant=fp8 → $0.158/1M tokens (ttft 168.9 ms, tpot 8.7 ms)

_Stopped: agent stop: fp8 gave best cost/throughput; further gains unlikely in search space_
