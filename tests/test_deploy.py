import pytest
from tuner.config import Config
from tuner.deploy import deploy, teardown, DeployError, CliResult


class FakeRunner:
    """Records every call; returns a queued result per call (or a single fixed one)."""
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def run(self, args):
        self.calls.append(args)
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]


def test_deploy_builds_container_args_and_parses_url():
    # real create/get shape: reachable address is status.public_endpoints[0] = "IP:PORT"
    runner = FakeRunner(CliResult(
        stdout='{"metadata":{"id":"aiendpoint-123"},"status":{"public_endpoints":["195.242.28.162:8000"]}}',
        stderr="", returncode=0))
    url = deploy(Config(model="Qwen/Qwen2.5-1.5B-Instruct", quantization="fp8"),
                 runner, token="secret-token", subnet_id="subnet-1")
    assert url == "http://195.242.28.162:8000/v1"
    args = runner.calls[0]
    assert "create" in args
    assert "--image" in args            # container-based deploy
    assert "--public" in args
    assert "--token" in args and "secret-token" in args
    assert "--subnet-id" in args and "subnet-1" in args
    # model + serving config travel inside the single --args string
    vllm_args = args[args.index("--args") + 1]
    assert "Qwen/Qwen2.5-1.5B-Instruct" in vllm_args
    assert "fp8" in vllm_args
    assert "--port 8000" in vllm_args


def test_deploy_raises_on_nonzero():
    runner = FakeRunner(CliResult(stdout="", stderr="boom", returncode=1))
    with pytest.raises(DeployError):
        deploy(Config(model="m"), runner, token="t")


def test_teardown_resolves_name_to_id_then_deletes():
    runner = FakeRunner(
        CliResult(stdout='{"metadata":{"id":"endpoint-123"}}', stderr="", returncode=0),  # get-by-name
        CliResult(stdout="", stderr="", returncode=0),                                     # delete
    )
    teardown("tuner-endpoint", runner)
    assert runner.calls[0][:4] == ["nebius", "ai", "endpoint", "get-by-name"]
    delete_call = runner.calls[1]
    assert delete_call[:4] == ["nebius", "ai", "endpoint", "delete"]
    assert "--id" in delete_call and "endpoint-123" in delete_call
