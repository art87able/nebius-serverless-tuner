from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Protocol
from tuner.config import Config

# A Serverless Endpoint is a container deployment: a vLLM image on a GPU platform.
# The model + vLLM serving config are passed through the container's --args, NOT as
# dedicated CLI flags (verified against `nebius ai endpoint create` v0.12.x).
DEFAULT_IMAGE = "vllm/vllm-openai:latest"   # pin to a Nebius-recommended tag during the live run
DEFAULT_PLATFORM = "gpu-l40s-d"
DEFAULT_PRESET = "1gpu-16vcpu-200gb"
CONTAINER_PORT = 8000


@dataclass(frozen=True)
class CliResult:
    stdout: str
    stderr: str
    returncode: int


class CliRunner(Protocol):
    def run(self, args: list[str]) -> CliResult: ...


class DeployError(RuntimeError):
    pass


def _vllm_args(config: Config) -> str:
    parts = ["--model", config.model,
             "--dtype", config.dtype,
             "--max-num-seqs", str(config.max_num_seqs),
             "--max-model-len", str(config.max_model_len)]
    if config.quantization:
        parts += ["--quantization", config.quantization]
    return " ".join(parts)


def _create_args(config: Config, name: str, image: str, platform: str,
                 preset: str, parent_id: str | None) -> list[str]:
    args = [
        "nebius", "ai", "endpoint", "create",
        "--name", name,
        "--image", image,
        "--platform", platform,
        "--preset", preset,
        "--container-port", str(CONTAINER_PORT),
        "--public",
        "--auth", "token",
        "--args", _vllm_args(config),
        "--format", "json",
    ]
    if parent_id:
        args += ["--parent-id", parent_id]
    return args


def _parse_url(stdout: str) -> str:
    """Pull the public endpoint URL out of the create response."""
    try:
        data = json.loads(stdout)
        for path in (("status", "url"), ("status", "endpoint_url"), ("metadata", "url")):
            node = data
            for key in path:
                node = node.get(key, {}) if isinstance(node, dict) else {}
            if isinstance(node, str) and node:
                return node
    except json.JSONDecodeError:
        pass
    m = re.search(r"https://\S+", stdout)
    if not m:
        raise DeployError(f"no endpoint URL in CLI output: {stdout!r}")
    return m.group(0)


def _parse_id(stdout: str) -> str:
    try:
        data = json.loads(stdout)
        for path in (("metadata", "id"), ("id",)):
            node = data
            for key in path:
                node = node.get(key, {}) if isinstance(node, dict) else {}
            if isinstance(node, str) and node:
                return node
    except json.JSONDecodeError:
        pass
    raise DeployError(f"no endpoint id in CLI output: {stdout!r}")


def deploy(config: Config, runner: CliRunner, *, name: str = "tuner-endpoint",
           image: str = DEFAULT_IMAGE, platform: str = DEFAULT_PLATFORM,
           preset: str = DEFAULT_PRESET, parent_id: str | None = None) -> str:
    """Create the Serverless Endpoint and return its OpenAI-compatible base URL."""
    res = runner.run(_create_args(config, name, image, platform, preset, parent_id))
    if res.returncode != 0:
        raise DeployError(res.stderr or "endpoint create failed")
    return _parse_url(res.stdout)


def teardown(name: str, runner: CliRunner) -> None:
    """Delete the endpoint by name (delete takes --id, so resolve name->id first)."""
    got = runner.run(["nebius", "ai", "endpoint", "get-by-name", "--name", name, "--format", "json"])
    if got.returncode != 0:
        raise DeployError(got.stderr or "endpoint get-by-name failed")
    endpoint_id = _parse_id(got.stdout)
    res = runner.run(["nebius", "ai", "endpoint", "delete", "--id", endpoint_id])
    if res.returncode != 0:
        raise DeployError(res.stderr or "endpoint delete failed")
