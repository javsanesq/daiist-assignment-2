import pytest
import warnings

from tools import setup_key


def test_key_helper_saves_without_echo(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(setup_key, "ROOT", tmp_path)
    monkeypatch.setattr(setup_key, "getpass", lambda _: "test-only-placeholder")
    setup_key.main()
    assert (tmp_path / ".env").read_text() == "ROBOFLOW_API_KEY=test-only-placeholder\n"
    assert "test-only-placeholder" not in capsys.readouterr().out


def test_key_helper_preserves_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_key, "ROOT", tmp_path)
    (tmp_path / ".env").write_text("existing configuration\n")
    monkeypatch.setattr(setup_key, "getpass", lambda _: (_ for _ in ()).throw(AssertionError("Must not ask")))
    setup_key.main()
    assert (tmp_path / ".env").read_text() == "existing configuration\n"


def test_key_helper_refuses_visible_input_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_key, "ROOT", tmp_path)
    def no_hidden_input(_):
        warnings.warn("No hidden terminal", setup_key.GetPassWarning)
        raise AssertionError("Must stop before fallback reads a key")
    monkeypatch.setattr(setup_key, "getpass", no_hidden_input)
    with pytest.raises(SystemExit, match="interactive terminal"):
        setup_key.main()
    assert not (tmp_path / ".env").exists()
