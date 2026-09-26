from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import hashlib
from pathlib import Path

DEFAULT_URL = "https://www.youtube.com/watch?v=6H6re5ECEZQ"

_TIMESTAMP_LINE_RE = re.compile(
    r"(?P<ts>(?:\d{1,2}:)?\d{1,2}:\d{2})"  # 0:00 or 00:00:00
    r"(?:\s*[-–—|:]+\s*|\s+)"  # common separators
    r"(?P<title>.+?)\s*$"
)


def require_program(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Missing required program: {name}. Install it and add it to PATH.")


def run(command: list[str], capture: bool = False) -> str:
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=None,
    )
    return result.stdout if capture else ""


def run_capture(command: list[str]) -> tuple[str, str]:
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout, result.stderr


def safe_name(value: str, *, max_length: int = 80) -> str:
    """Sanitize a string for use in file/folder names.

    Also truncates to keep Windows path lengths sane.
    """

    value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "", str(value or ""))
    value = re.sub(r"\s+", " ", value).strip().rstrip(".")
    value = value or "Untitled"

    # Keep names short to avoid deep-path issues on Windows and overly long zips.
    if max_length and len(value) > max_length:
        digest = hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:8]
        keep = max(1, max_length - (len(digest) + 2))
        value = f"{value[:keep].rstrip()}~{digest}"
    return value


def smart_title_from_info(info: dict, url: str | None = None) -> str:
    """Best-effort title detection when yt-dlp doesn't provide `title`.

    Prefers human-readable metadata when present; falls back to uploader/id/url.
    """

    def first_str(*values: object) -> str | None:
        for v in values:
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    # Common yt-dlp fields (varies by extractor).
    title = first_str(
        info.get("title"),
        info.get("fulltitle"),
        info.get("track"),
        info.get("alt_title"),
    )
    if title:
        artist = first_str(info.get("artist"), info.get("creator"), info.get("uploader"))
        if artist and title and artist.lower() not in title.lower():
            return safe_name(f"{artist} - {title}")
        return safe_name(title)

    uploader = first_str(info.get("uploader"), info.get("channel"), info.get("creator"))
    vid = first_str(info.get("id"), info.get("display_id"), info.get("webpage_url_basename"))
    if uploader and vid:
        return safe_name(f"{uploader} - {vid}")
    if uploader:
        return safe_name(uploader)
    if vid:
        return safe_name(vid)
    if url:
        return safe_name(url)
    return "YouTube Audio"


_GENERIC_TRACK_NAME_RE = re.compile(
    r"^\s*(?:\d{1,3}\s*[-._)]\s*)?(?:track|song)\s*\d{1,3}\s*$",
    re.IGNORECASE,
)


def _synchsafe_to_int(value: bytes) -> int:
    return (
        ((value[0] & 0x7F) << 21)
        | ((value[1] & 0x7F) << 14)
        | ((value[2] & 0x7F) << 7)
        | (value[3] & 0x7F)
    )


def _read_id3v2_text_frames(path: Path) -> dict[str, str]:
    """Extract a small set of ID3v2 text frames (best-effort, no deps).

    Only supports ID3v2.3/2.4 (common for mp3). Returns lowercased keys.
    """

    try:
        data = path.read_bytes()
    except OSError:
        return {}

    if len(data) < 10 or data[:3] != b"ID3":
        return {}

    major = data[3]
    if major not in (3, 4):
        return {}

    tag_size = _synchsafe_to_int(data[6:10])
    tag = data[10 : 10 + tag_size]

    def decode_text(payload: bytes) -> str:
        if not payload:
            return ""
        enc = payload[0]
        raw = payload[1:]
        if enc == 0:
            return raw.split(b"\x00", 1)[0].decode("latin1", errors="replace").strip()
        if enc == 1:
            return raw.decode("utf-16", errors="replace").strip("\x00").strip()
        if enc == 2:
            return raw.decode("utf-16-be", errors="replace").strip("\x00").strip()
        if enc == 3:
            return raw.decode("utf-8", errors="replace").strip("\x00").strip()
        return raw.decode("latin1", errors="replace").strip()

    frames: dict[str, str] = {}
    offset = 0
    while offset + 10 <= len(tag):
        frame_id = tag[offset : offset + 4]
        if frame_id == b"\x00\x00\x00\x00":
            break
        size_bytes = tag[offset + 4 : offset + 8]
        frame_size = _synchsafe_to_int(size_bytes) if major == 4 else int.from_bytes(size_bytes, "big")
        payload = tag[offset + 10 : offset + 10 + frame_size]
        offset += 10 + frame_size

        try:
            fid = frame_id.decode("ascii")
        except UnicodeDecodeError:
            continue
        if not fid.startswith("T") or fid in ("TXXX",):
            continue
        value = decode_text(payload)
        if value:
            frames[fid.lower()] = value

    return frames


def smart_title_from_media_path(path: Path) -> str:
    """Best-effort title for local media files.

    Priority:
    1) Embedded metadata (ID3 title/artist/album)
    2) A non-generic parent/grandparent folder name (handles re-processing already-split tracks)
    3) Filename stem
    """

    frames = _read_id3v2_text_frames(path)
    title = (frames.get("tit2") or "").strip()
    artist = (frames.get("tpe1") or frames.get("tpe2") or "").strip()
    album = (frames.get("talb") or "").strip()

    if title:
        if artist and artist.lower() not in title.lower():
            return safe_name(f"{artist} - {title}")
        return safe_name(title)
    if album and artist and artist.lower() not in album.lower():
        return safe_name(f"{artist} - {album}")
    if album:
        return safe_name(album)

    # Path heuristic: walk up a few levels and pick the first non-generic name.
    stem = (path.stem or "").strip()
    parent = (path.parent.name or "").strip()
    candidates: list[str] = [c for c in (stem, parent if parent != stem else "") if c]
    if path.parent.parent and path.parent.parent.name:
        candidates.append(path.parent.parent.name.strip())
    if path.parent.parent and path.parent.parent.parent and path.parent.parent.parent.name:
        candidates.append(path.parent.parent.parent.name.strip())

    for candidate in candidates:
        if not candidate:
            continue
        if _GENERIC_TRACK_NAME_RE.match(candidate):
            continue
        if candidate.lower() in {"songs", "music", "downloads"}:
            continue
        return safe_name(candidate)

    return safe_name(stem or parent or "Untitled")


def _timestamp_to_seconds(value: str) -> float | None:
    parts = value.strip().split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        minutes, seconds = nums
        hours = 0
    else:
        hours, minutes, seconds = nums
    if minutes < 0 or seconds < 0 or seconds >= 60:
        return None
    return float(hours * 3600 + minutes * 60 + seconds)


def chapters_from_description(description: str, duration: float | None = None) -> list[dict]:
    """Best-effort chapter extraction from typical YouTube tracklists in descriptions."""
    chapters: list[dict] = []
    starts: list[float] = []
    for raw_line in (description or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _TIMESTAMP_LINE_RE.search(line)
        if not match:
            continue
        start = _timestamp_to_seconds(match.group("ts"))
        if start is None:
            continue
        title = match.group("title").strip()
        if not title:
            continue
        chapters.append({"title": title, "start_time": start, "end_time": None})
        starts.append(start)

    # Fill end times from the next start time or duration.
    chapters = sorted(chapters, key=lambda c: float(c.get("start_time") or 0))
    normalized: list[dict] = []
    seen_starts: set[float] = set()
    for index, chapter in enumerate(chapters):
        start = float(chapter.get("start_time") or 0)
        if start in seen_starts:
            continue
        seen_starts.add(start)
        end: float | None
        if index + 1 < len(chapters):
            end = float(chapters[index + 1].get("start_time") or 0)
        else:
            end = float(duration) if duration else None
        normalized.append({"title": chapter.get("title") or f"Song {index + 1:02d}", "start_time": start, "end_time": end})
    return normalized


def track_titles_from_text(tracklist: str) -> list[str]:
    """Return newline-separated titles when the input has no timestamps."""
    lines = [line.strip() for line in (tracklist or "").splitlines() if line.strip()]
    if not lines or any(_TIMESTAMP_LINE_RE.search(line) for line in lines):
        return []
    return lines


def apply_track_titles(chapters: list[dict], titles: list[str]) -> list[dict]:
    """Apply user-provided titles to chapters while retaining detected timings."""
    if not titles:
        return chapters
    for index, chapter in enumerate(chapters):
        if index < len(titles):
            chapter["title"] = titles[index]
    return chapters


def normalize_chapters(chapters: list[dict], duration: float | None = None) -> list[dict]:
    """Normalize chapter dicts to have float start/end times; infer missing ends when possible."""
    if not chapters:
        return []
    cleaned: list[dict] = []
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        start_raw = chapter.get("start_time")
        if start_raw is None:
            continue
        try:
            start = float(start_raw)
        except (TypeError, ValueError):
            continue
        title = chapter.get("title") or ""
        end_raw = chapter.get("end_time")
        end: float | None
        if end_raw is None:
            end = None
        else:
            try:
                end = float(end_raw)
            except (TypeError, ValueError):
                end = None
        cleaned.append({"title": title, "start_time": start, "end_time": end})

    cleaned.sort(key=lambda c: float(c.get("start_time") or 0))
    for i, chapter in enumerate(cleaned):
        start = float(chapter.get("start_time") or 0)
        end = chapter.get("end_time")
        if end is None or end <= start:
            if i + 1 < len(cleaned):
                next_start = float(cleaned[i + 1].get("start_time") or 0)
                chapter["end_time"] = next_start if next_start > start else None
            elif duration and duration > start:
                chapter["end_time"] = float(duration)

    return [c for c in cleaned if c.get("start_time") is not None]


def chapters_from_silence(
    source: Path,
    duration: float | None,
    *,
    threshold_db: float = -35.0,
    min_silence_duration: float = 0.8,
    min_track_duration: float = 30.0,
) -> list[dict]:
    # Force audio-only processing and downsample before silencedetect.
    # This is a heuristic pass; lower fidelity is fine and usually much faster.
    _, stderr = run_capture([
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-af",
        "aformat=channel_layouts=mono,aresample=8000,"
        + f"silencedetect=n={threshold_db}dB:d={min_silence_duration}",
        "-f",
        "null",
        "-",
    ])

    silence_starts: list[float] = []
    silence_ends: list[float] = []
    for line in stderr.splitlines():
        line = line.strip()
        if "silence_start:" in line:
            try:
                silence_starts.append(float(line.split("silence_start:", 1)[1].strip().split()[0]))
            except ValueError:
                pass
        elif "silence_end:" in line:
            try:
                silence_ends.append(float(line.split("silence_end:", 1)[1].strip().split()[0]))
            except ValueError:
                pass

    boundaries: list[float] = []
    for start, end in zip(silence_starts, silence_ends):
        if end <= start:
            continue
        boundaries.append((start + end) / 2.0)

    boundaries = [b for b in boundaries if b > 0]
    boundaries = sorted(set(boundaries))

    chapters: list[dict] = []
    prev = 0.0
    for boundary in boundaries:
        if boundary - prev >= min_track_duration:
            chapters.append({"title": "", "start_time": prev, "end_time": boundary})
            prev = boundary
    if duration and duration - prev >= min_track_duration:
        chapters.append({"title": "", "start_time": prev, "end_time": duration})
    elif not chapters:
        return []

    for idx, chapter in enumerate(chapters, 1):
        chapter["title"] = chapter.get("title") or f"Track {idx:02d}"
    return chapters


def get_video_info(url: str) -> dict:
    raw = run(["yt-dlp", "--no-warnings", "--dump-single-json", "--skip-download", url], True)
    return json.loads(raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract YouTube audio and split chaptered music videos into song folders."
    )
    parser.add_argument("url", nargs="?", default=DEFAULT_URL)
    parser.add_argument("-o", "--output", default="songs", help="Output directory")
    parser.add_argument("--format", choices=["m4a", "mp3", "wav"], default="mp3")
    parser.add_argument(
        "--auto-silence",
        action="store_true",
        help="If no chapters/tracklist are found, split by detected silences.",
    )
    parser.add_argument("--silence-threshold-db", type=float, default=-35.0)
    parser.add_argument("--min-silence", type=float, default=0.8)
    parser.add_argument("--min-track", type=float, default=30.0)
    args = parser.parse_args()

    require_program("yt-dlp")
    require_program("ffmpeg")

    info = get_video_info(args.url)
    video_title = safe_name(info.get("title", "YouTube Audio"))
    duration = info.get("duration")
    duration = float(duration) if duration is not None else None

    chapters = info.get("chapters") or []
    if not chapters:
        chapters = chapters_from_description(info.get("description") or "", duration)
    chapters = normalize_chapters(chapters, duration)

    root = Path(args.output).expanduser().resolve() / video_title
    root.mkdir(parents=True, exist_ok=True)
    source = root / "_full_audio.m4a"

    print(f"Downloading audio: {video_title}")
    run([
        "yt-dlp",
        "--no-playlist",
        "-f",
        "bestaudio/best",
        "-x",
        "--audio-format",
        "m4a",
        "-o",
        str(source),
        args.url,
    ])

    if not chapters and args.auto_silence:
        chapters = normalize_chapters(
            chapters_from_silence(
                source,
                duration,
                threshold_db=args.silence_threshold_db,
                min_silence_duration=args.min_silence,
                min_track_duration=args.min_track,
            ),
            duration,
        )

    if not chapters:
        raise SystemExit(
            "No chapters were found. Add YouTube chapters, include a timestamped tracklist in the description, or rerun with --auto-silence."
        )

    extension = args.format
    codec = {"m4a": "aac", "mp3": "libmp3lame", "wav": "pcm_s16le"}[extension]
    for index, chapter in enumerate(chapters, 1):
        title = safe_name(chapter.get("title", f"Song {index:02d}"))
        start = float(chapter.get("start_time", 0))
        end = chapter.get("end_time")
        if end is not None:
            end = float(end)
        if end is not None and end <= start:
            continue

        output = root / f"{title}.{extension}"
        command = ["ffmpeg", "-hide_banner", "-y", "-ss", str(start)]
        if end is not None:
            command += ["-to", str(end)]
        command += ["-i", str(source), "-map", "0:a:0", "-vn", "-c:a", codec]
        if extension == "m4a":
            command += ["-b:a", "192k"]
        elif extension == "mp3":
            command += ["-q:a", "2"]
        command.append(str(output))
        print(f"[{index}/{len(chapters)}] {title}")
        run(command)

    source.unlink(missing_ok=True)
    print(f"Saved songs to: {root}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
