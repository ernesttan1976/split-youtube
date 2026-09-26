from __future__ import annotations

import os
import sys
from pathlib import Path

from split_youtube_songs import acoustid_lookup_title, load_dotenv


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python recognize_audio.py <path-to-audio>")
        return 2

    path = Path(argv[1])
    if not path.exists():
        print(f"Not found: {path}")
        return 2

    if not os.environ.get("ACOUSTID_API_KEY"):
        load_dotenv()
    api_key = os.environ.get("ACOUSTID_API_KEY", "")
    if not api_key.strip():
        print("Missing env var: ACOUSTID_API_KEY")
        return 2

    try:
        title = acoustid_lookup_title(path, api_key=api_key)
    except FileNotFoundError:
        print("Missing dependency: fpcalc (Chromaprint). Install it and ensure it's on PATH.")
        return 2

    print(title or "(no match)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
