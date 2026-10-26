import pytest
import download_examples as module


class Response:
    def __init__(self, status=206, content_range="bytes 0-3/10", data=b"abcd"):
        self.status, self.headers, self.data = status, {"Content-Range": content_range}, data
        self.read_called = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, count):
        self.read_called = True
        return self.data[:count]


@pytest.mark.parametrize("status,header", [(200, "bytes 0-3/10"), (206, "bytes 1-4/10")])
def test_refuses_full_or_wrong_range_without_reading(monkeypatch, status, header):
    response = Response(status, header)
    monkeypatch.setattr(module, "urlopen", lambda *args, **kwargs: response)
    with pytest.raises(ValueError):
        module.RemoteZip("https://example.invalid", 10).read(4)
    assert not response.read_called


def test_range_budget_and_truncation(monkeypatch):
    remote = module.RemoteZip("https://example.invalid", 10, budget=3)
    with pytest.raises(ValueError):
        remote.read(4)
    monkeypatch.setattr(module, "urlopen", lambda *args, **kwargs: Response(data=b"abc"))
    with pytest.raises(ValueError):
        module.RemoteZip("https://example.invalid", 10).read(4)


def test_verified_range_advances_position(monkeypatch):
    monkeypatch.setattr(module, "urlopen", lambda *args, **kwargs: Response())
    remote = module.RemoteZip("https://example.invalid", 10)
    assert remote.read(4) == b"abcd"
    assert remote.tell() == remote.bytes_read == 4
    assert remote.seek(-2, 2) == 8


def test_network_errors_do_not_expose_credentials(monkeypatch, tmp_path):
    def absent(*args):
        raise FileNotFoundError
    monkeypatch.setattr(module, "load_demo_assets", absent)
    monkeypatch.setattr(module, "load_api_key", lambda **kwargs: "SECRET_TEST_KEY")
    monkeypatch.setattr(module, "_context", lambda root: ({"dataset": {"workspace": "w", "project": "p", "version": 3}}, None))
    def failure(*args, **kwargs):
        raise RuntimeError("https://example.invalid?api_key=SECRET_TEST_KEY")
    monkeypatch.setattr(module, "urlopen", failure)
    with pytest.raises(ValueError) as captured:
        module.download_examples(tmp_path)
    assert "SECRET" not in str(captured.value)
    assert captured.value.__suppress_context__
