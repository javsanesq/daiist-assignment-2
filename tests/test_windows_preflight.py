import pytest

from tools.windows_preflight import require_windows


def test_training_platform_guard():
    require_windows("Windows")
    with pytest.raises(RuntimeError, match="Mac"):
        require_windows("Darwin")
    with pytest.raises(RuntimeError):
        require_windows("Linux")
