from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Protocol
from tuner.config import Config


@dataclass(frozen=True)
class CliResult:
    stdout: str
    stderr: str
    returncode: int


class CliRunner(Protocol):
    def run(self, args: list[str]) -> CliResult: ...


class DeployError(RuntimeError):
    pass


def _endpoint_args(config: Config, name: str) -> list[str]:
    args = [
        "nebius", "ai", "endpoint", "create",
        "--name", name,
        "--model", config.model,
        "--dtype", config.dtype,
        "--max-num-seqs", str(config.max_num_seqs),
        "--max-model-len", str(config.max_model_len),
        "--format", "json",
    ]
    if config.quantization:
        args += ["--quantization", config.quantization]
    return args


def _parse_url(stdout: str) -> str:
    try:
        return json.loads(stdout)["url"]
    except (json.JSONDecodeError, KeyError):
        m = re.search(r"https://\S+", stdout)
        if not m:
            raise DeployError(f"no endpoint URL in CLI output: {stdout!r}")
        return m.group(0)


def deploy(config: Config, runner: CliRunner, name: str = "tuner-endpoint") -> str:
    res = runner.run(_endpoint_args(config, name))
    if res.returncode != 0:
        raise DeployError(res.stderr or "endpoint create failed")
    return _parse_url(res.stdout)


def teardown(name: str, runner: CliRunner) -> None:
    res = runner.run(["nebius", "ai", "endpoint", "delete", "--name", name])
    if res.returncode != 0:
        raise DeployError(res.stderr or "endpoint delete failed")
