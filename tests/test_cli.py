from tuner import cli
from tuner.config import Config, Metrics, Stop


def test_cli_run_wires_seams_and_tears_down(monkeypatch, capsys):
    # patch the live seams so the CLI runs fully offline
    torn = []
    monkeypatch.setattr(cli, "make_deploy_fn", lambda token, holder=None: (lambda cfg: "https://e/v1"))
    monkeypatch.setattr(cli, "make_bench_fn", lambda sweep, token: (lambda url: Metrics(50, 10, 500, 1000, 1.0, 8, 16)))
    monkeypatch.setattr(cli, "make_agent_fn", lambda *a, **k: (lambda hist, space: Stop("done")))
    monkeypatch.setattr(cli, "make_teardown_fn", lambda: (lambda: torn.append(True)))
    rc = cli.main(["tune", "--model", "Qwen/Qwen2.5-1.5B-Instruct",
                   "--max-iters", "1", "--budget-usd", "5", "--gpu-rate", "2.0",
                   "--gpu", "a100"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Winning config" in out
    assert "Qwen/Qwen2.5-1.5B-Instruct" in out
    assert torn == [True]          # endpoint always released


def test_cli_tears_down_even_on_error(monkeypatch):
    torn = []
    def boom(cfg):
        raise RuntimeError("deploy exploded")
    monkeypatch.setattr(cli, "make_deploy_fn", lambda token, holder=None: boom)
    monkeypatch.setattr(cli, "make_bench_fn", lambda sweep, token: (lambda url: None))
    monkeypatch.setattr(cli, "make_agent_fn", lambda *a, **k: (lambda hist, space: Stop("x")))
    monkeypatch.setattr(cli, "make_teardown_fn", lambda: (lambda: torn.append(True)))
    import pytest
    with pytest.raises(RuntimeError):
        cli.main(["tune", "--model", "m", "--max-iters", "1",
                  "--budget-usd", "5", "--gpu-rate", "2.0"])
    assert torn == [True]          # teardown still runs on failure


def test_endpoint_as_agent_brain_uses_deployed_url(monkeypatch):
    # AGENT_LLM_BASE_URL=endpoint routes agent calls to the tuner's own endpoint,
    # late-bound to whatever URL the most recent deploy produced.
    from tuner import adapters
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((url, headers, json))
        class R:
            def raise_for_status(self): pass
            def json(self):
                return {"choices": [{"message": {"content": '{"action":"stop","rationale":"ok"}'}}]}
        return R()

    monkeypatch.setattr(adapters.requests, "post", fake_post)
    monkeypatch.setenv("AGENT_LLM_BASE_URL", "endpoint")
    monkeypatch.setenv("ENDPOINT_MODEL", "test-model")

    holder = {"url": None}
    agent_fn = cli.make_agent_fn("tok123", holder)
    holder["url"] = "http://1.2.3.4/v1"       # set by deploy in the real flow

    from tuner.config import Iteration, Metrics as M, SearchSpace
    from tuner.pricing import cost
    it = Iteration(Config(model="m"), M(50, 10, 500, 1000, 1.0, 8, 16),
                   cost(M(50, 10, 500, 1000, 1.0, 8, 16), 2.0))
    decision = agent_fn([it], SearchSpace(("auto",), (64,), (None,)))

    assert isinstance(decision, Stop)
    url, headers, body = calls[0]
    assert url == "http://1.2.3.4/v1/chat/completions"
    assert headers["Authorization"] == "Bearer tok123"
    assert body["model"] == "test-model"


def test_endpoint_brain_fails_clearly_before_first_deploy(monkeypatch):
    from tuner import adapters
    monkeypatch.setenv("AGENT_LLM_BASE_URL", "endpoint")
    gen = adapters.endpoint_llm_generate(lambda: None, "t", "m")
    import pytest
    with pytest.raises(RuntimeError, match="before any endpoint"):
        gen("hi")
