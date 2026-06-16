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
