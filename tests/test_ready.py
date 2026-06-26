import pytest
from tuner.adapters import wait_for_ready


class FakeResp:
    def __init__(self, status_code):
        self.status_code = status_code


class FakeGet:
    """Returns queued responses; raises if a queued item is an Exception.

    A single queued item is returned repeatedly (matches FakeRunner's style).
    """
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, **kw):
        self.calls.append((url, kw))
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def test_ready_returns_when_models_endpoint_200():
    get = FakeGet(FakeResp(200))
    clock = FakeClock()
    wait_for_ready("http://ip:8000/v1", "tok",
                   get_fn=get, sleep_fn=clock.sleep, now_fn=clock)
    assert get.calls[0][0] == "http://ip:8000/v1/models"
    assert get.calls[0][1]["headers"]["Authorization"] == "Bearer tok"


def test_ready_polls_through_503_and_connection_errors():
    get = FakeGet(FakeResp(503), ConnectionError("refused"), FakeResp(200))
    clock = FakeClock()
    wait_for_ready("http://ip:8000/v1", "tok", interval_s=5,
                   get_fn=get, sleep_fn=clock.sleep, now_fn=clock)
    assert len(get.calls) == 3


def test_ready_times_out_if_never_ready():
    get = FakeGet(FakeResp(503))
    clock = FakeClock()
    with pytest.raises(TimeoutError):
        wait_for_ready("http://ip:8000/v1", "tok", timeout_s=20, interval_s=10,
                       get_fn=get, sleep_fn=clock.sleep, now_fn=clock)
