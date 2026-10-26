"""Save a Roboflow key locally without echoing it. Run from an interactive terminal."""
from __future__ import annotations

from getpass import getpass, GetPassWarning
from pathlib import Path
import warnings

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    target = ROOT / ".env"
    if target.exists():
        print("A .env file already exists. It was not changed. Verify it with: uv run python api_key.py")
        return
    # Never let getpass fall back to echoing a key in a noninteractive tool log.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", GetPassWarning)
            key = getpass("Paste your Roboflow private API key (input will stay hidden), then press Enter: ").strip()
    except (GetPassWarning, EOFError):
        raise SystemExit("Open an interactive terminal and rerun this command; hidden input is unavailable here.")
    if not key or any(c in key for c in "\r\n \t\"'"):
        raise SystemExit("The key is empty or contains unexpected whitespace/quotes; nothing was saved.")
    # Exclusive creation avoids overwriting a key added while the prompt was open.
    with target.open("x", encoding="utf-8") as handle:
        handle.write(f"ROBOFLOW_API_KEY={key}\n")
    target.chmod(0o600)
    print("Saved to the ignored .env file. Next: uv run python api_key.py")


if __name__ == "__main__":
    main()
