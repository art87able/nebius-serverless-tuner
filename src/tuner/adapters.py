from __future__ import annotations
import os
import subprocess
import time
import requests
from tuner.config import RequestResult
from tuner.deploy import CliResult


class SubprocessRunner:
    def run(self, args: list[str]) -> CliResult:
        p = subprocess.run(args, capture_output=True, text=True)
        return CliResult(stdout=p.stdout, stderr=p.stderr, returncode=p.returncode)


def get_subnet_id(runner) -> str:
    """Resolve the tenant's first subnet id (needed by `endpoint create --subnet-id`)."""
    res = runner.run(["nebius", "vpc", "subnet", "list",
                      "--format", "jsonpath={.items[0].metadata.id}"])
    if res.returncode != 0:
        raise RuntimeError(res.stderr or "could not list subnets")
    return res.stdout.strip()


class HttpClient:
    """OpenAI-compatible chat client against a deployed endpoint."""
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def chat(self, input_tokens: int, output_tokens: int) -> RequestResult:
        prompt = "word " * input_tokens
        start = time.monotonic()
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": output_tokens, "temperature": 0},
            timeout=120,
        )
        total_ms = (time.monotonic() - start) * 1000
        r.raise_for_status()
        body = r.json()
        out_tok = body.get("usage", {}).get("completion_tokens", output_tokens)
        # vLLM returns no per-token TTFT without streaming; approximate TTFT as
        # total - (out_tok-1)*observed-tpot is circular, so for the non-stream path
        # record ttft == total_ms (documented limitation; streaming variant is future work).
        return RequestResult(ttft_ms=round(total_ms, 2), total_ms=round(total_ms, 2),
                             output_tokens=out_tok)


def llm_generate(prompt: str) -> str:
    """The agent's brain — uses Nebius Token Factory (cheap) by default."""
    base = os.environ.get("AGENT_LLM_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
    key = os.environ["NEBIUS_API_KEY"]
    model = os.environ.get("AGENT_LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
    r = requests.post(f"{base.rstrip('/')}/chat/completions",
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": model, "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 256, "temperature": 0}, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
