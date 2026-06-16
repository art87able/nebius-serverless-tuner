from tuner import cli
from tuner.config import Config, Metrics, Stop


def test_cli_run_wires_seams(monkeypatch, capsys):
    # patch the live seams so the CLI runs fully offline
    monkeypatch.setattr(cli, "make_deploy_fn", lambda: (lambda cfg: "https://e/v1"))
    monkeypatch.setattr(cli, "make_bench_fn", lambda url_holder: (lambda url: Metrics(50, 10, 500, 1000, 1.0, 8, 16)))
    monkeypatch.setattr(cli, "make_agent_fn", lambda: (lambda hist, space: Stop("done")))
    rc = cli.main(["tune", "--model", "Qwen/Qwen2.5-1.5B-Instruct",
                   "--max-iters", "1", "--budget-usd", "5", "--gpu-rate", "2.0",
                   "--gpu", "a100"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Winning config" in out
    assert "Qwen/Qwen2.5-1.5B-Instruct" in out
