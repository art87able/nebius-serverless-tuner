import pytest
from tuner.config import Config
from tuner.deploy import deploy, teardown, DeployError, CliResult


class FakeRunner:
    def __init__(self, result):
        self.result = result
        self.last_args = None

    def run(self, args):
        self.last_args = args
        return self.result


def test_deploy_builds_args_and_parses_url():
    runner = FakeRunner(CliResult(
        stdout='{"status":"running","url":"https://abc.endpoints.nebius.com/v1"}',
        stderr="", returncode=0))
    url = deploy(Config(model="Qwen/Qwen2.5-1.5B-Instruct", quantization="fp8"), runner)
    assert url == "https://abc.endpoints.nebius.com/v1"
    assert "create" in runner.last_args
    assert "Qwen/Qwen2.5-1.5B-Instruct" in runner.last_args
    assert "fp8" in runner.last_args


def test_deploy_raises_on_nonzero():
    runner = FakeRunner(CliResult(stdout="", stderr="boom", returncode=1))
    with pytest.raises(DeployError):
        deploy(Config(model="m"), runner)
