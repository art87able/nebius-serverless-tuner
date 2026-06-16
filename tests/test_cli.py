from tuner import cli
from tuner.config import Config, Metrics, Stop


def test_cli_run_wires_seams_and_tears_down(monkeypatch, capsys):
    # patch the live seams so the CLI runs fully offline
    torn = []
    monkeypatch.setattr(cli, "make_deploy_fn", lambda token: (lambda cfg: "https://e/v1"))
    monkeypatch.setattr(cli, "make_bench_fn", lambda sweep, token: (lambda url: Metrics(50, 10, 500, 1000, 1.0, 8, 16)))
    monkeypatch.setattr(cli, "make_agent_fn", lambda: (lambda hist, space: Stop("done")))
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
    monkeypatch.setattr(cli, "make_deploy_fn", lambda token: boom)
    monkeypatch.setattr(cli, "make_bench_fn", lambda sweep, token: (lambda url: None))
    monkeypatch.setattr(cli, "make_agent_fn", lambda: (lambda hist, space: Stop("x")))
    monkeypatch.setattr(cli, "make_teardown_fn", lambda: (lambda: torn.append(True)))
    import pytest
    with pytest.raises(RuntimeError):
        cli.main(["tune", "--model", "m", "--max-iters", "1",
                  "--budget-usd", "5", "--gpu-rate", "2.0"])
    assert torn == [True]          # teardown still runs on failure
