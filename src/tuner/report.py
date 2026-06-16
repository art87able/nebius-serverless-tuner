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
