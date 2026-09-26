from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import gradio as gr

from split_youtube_songs import (
    DEFAULT_URL,
    chapters_from_description,
    normalize_chapters,
    safe_name,
    smart_title_from_info,
    smart_title_from_media_path,
)


def command(args: list[str], capture: bool = False) -> str:
    result = subprocess.run(
        args,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    return result.stdout if capture else ""


def command_capture(args: list[str]) -> tuple[str, str]:
    result = subprocess.run(
        args,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout, result.stderr


def chapters_from_file(source: Path) -> list[dict]:
    raw = command([
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_chapters",
        str(source),
    ], True)
    return json.loads(raw).get("chapters", [])


def duration_from_file(source: Path) -> float | None:
    raw = command([
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        str(source),
    ], True)
    duration = (json.loads(raw).get("format") or {}).get("duration")
    if duration is None:
        return None
    try:
        return float(duration)
    except (TypeError, ValueError):
        return None


def chapters_from_silence(
    source: Path,
    *,
    threshold_db: float = -35.0,
    min_silence_duration: float = 0.8,
    min_track_duration: float = 30.0,
) -> list[dict]:
    """Heuristic split: use ffmpeg silencedetect to find boundaries.

    Works best when tracks are separated by near-silence. Crossfades and
    constant-noise mixes will not split well.
    """

    # Force audio-only processing and downsample before silencedetect.
    # This is a heuristic pass; lower fidelity is fine and usually much faster.
    _, stderr = command_capture([
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

    duration = duration_from_file(source)
    boundaries = [b for b in boundaries if b > 0]
    boundaries = sorted(set(boundaries))

    chapters: list[dict] = []
    prev = 0.0
    for boundary in boundaries:
        if boundary - prev >= min_track_duration:
            chapters.append({"title": "", "start_time": prev, "end_time": boundary})
            prev = boundary
    if duration is None:
        # Let the final split run to EOF.
        if not chapters or (prev - chapters[-1]["start_time"]) >= min_track_duration:
            chapters.append({"title": "", "start_time": prev, "end_time": None})
    else:
        if duration - prev >= min_track_duration:
            chapters.append({"title": "", "start_time": prev, "end_time": duration})

    # Assign simple names.
    for idx, chapter in enumerate(chapters, 1):
        chapter["title"] = chapter.get("title") or f"Track {idx:02d}"
    return chapters


def split_source(
    source: Path,
    title: str,
    chapters: list[dict],
    output: Path,
    extension: str,
    *,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> Path:
    if not chapters:
        raise ValueError(
            "No chapters were found. Use a YouTube URL with chapters, paste a timestamped tracklist, "
            "or provide a local media file with embedded chapters."
        )

    root = output / safe_name(title)
    root.mkdir(parents=True, exist_ok=True)
    codec = {"m4a": "aac", "mp3": "libmp3lame", "wav": "pcm_s16le"}[extension]
    for index, chapter in enumerate(chapters, 1):
        raw_title = chapter.get("title") or f"Song {index:02d}"

        # Avoid double-prefixes like "01 - 01 - Track" when input titles already contain numbering.
        cleaned = str(raw_title).strip()
        cleaned = re.sub(rf"^\s*{index:02d}\s*[-.:)]\s*", "", cleaned)
        cleaned = re.sub(rf"^\s*{index}\s*[-.:)]\s*", "", cleaned)
        cleaned = re.sub(rf"^\s*track\s*0*{index}\s*[-.:)]?\s*", "", cleaned, flags=re.IGNORECASE)
        chapter_title = safe_name(cleaned or raw_title)
        start = float(chapter.get("start_time") or 0)
        end_raw = chapter.get("end_time")
        end = None if end_raw is None else float(end_raw)
        if end is not None and end <= start:
            continue

        # Prefix keeps ordering stable and avoids collisions when titles repeat or are truncated.
        base = safe_name(f"{index:02d} - {chapter_title}", max_length=80)
        destination = root / f"{base}.{extension}"
        if on_progress is not None:
            on_progress(index, len(chapters), base)
        args = ["ffmpeg", "-hide_banner", "-y", "-ss", str(start)]
        if end is not None:
            args += ["-to", str(end)]
        args += ["-i", str(source), "-map", "0:a:0", "-vn", "-c:a", codec]
        if extension == "m4a":
            args += ["-b:a", "192k"]
        elif extension == "mp3":
            args += ["-q:a", "2"]
        command(args + [str(destination)])
    return root


def process_stream(
    url: str,
    media_file: str | None,
    tracklist: str,
    auto_split: bool,
    silence_threshold_db: float,
    min_silence_duration: float,
    min_track_duration: float,
    output_format: str,
    progress=gr.Progress(),
):
    if not url.strip() and not media_file:
        yield None, "Provide a YouTube URL or choose a local audio/video file."
        return
    if url.strip() and media_file:
        yield None, "Choose one input source, not both."
        return

    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        output = work / "songs"
        output.mkdir()
        if url.strip():
            url = url.strip()
            progress(0, desc="Fetching metadata")
            info = json.loads(command([
                "yt-dlp", "--no-warnings", "--dump-single-json", "--skip-download", url
            ], True))
            title = smart_title_from_info(info, url)
            duration = info.get("duration")
            duration = float(duration) if duration is not None else None

            chapters = info.get("chapters") or []
            if not chapters:
                if tracklist.strip():
                    chapters = chapters_from_description(tracklist, duration)
                if not chapters:
                    chapters = chapters_from_description(info.get("description") or "", duration)
            chapters = normalize_chapters(chapters, duration)

            # If there are no chapters and the user did not opt into auto-splitting,
            # fail fast instead of downloading the full audio.
            if not chapters and not auto_split:
                yield (
                    None,
                    "No chapters/tracklist found. Paste timestamps in the tracklist box, "
                    "or enable 'Auto split by silence' (can be slow for long videos).",
                )
                return

            source = work / "source.m4a"
            progress(0.05, desc="Downloading audio")
            yield None, f"Downloading audio: {safe_name(title)}"
            command([
                "yt-dlp", "--no-playlist", "-f", "bestaudio/best", "-x",
                "--audio-format", "m4a", "-o", str(source), url
            ])
        else:
            source = Path(media_file)
            title = smart_title_from_media_path(source)
            chapters = normalize_chapters(chapters_from_file(source), None)
            if not chapters and tracklist.strip():
                chapters = normalize_chapters(chapters_from_description(tracklist, None), None)

        if not chapters and auto_split:
            progress(0.2, desc="Detecting silences")
            yield None, "No chapters found, auto-splitting by silence (this can take a while)."
            chapters = normalize_chapters(
                chapters_from_silence(
                    source,
                    threshold_db=silence_threshold_db,
                    min_silence_duration=min_silence_duration,
                    min_track_duration=min_track_duration,
                ),
                duration_from_file(source),
            )

        progress(0.3, desc="Splitting tracks")

        def on_split_progress(i: int, total: int, name: str) -> None:
            # Map split loop into 30%..95%.
            frac = 0.3 + (0.65 * (i / max(1, total)))
            progress(frac, desc=f"Splitting {i}/{total}")

        root = split_source(source, title, chapters, output, output_format, on_progress=on_split_progress)
        progress(0.96, desc="Creating zip")
        archive_base = work / safe_name(title)
        archive = Path(shutil.make_archive(str(archive_base), "zip", root.parent, root.name))
        destination = Path(tempfile.gettempdir()) / archive.name
        shutil.copy2(archive, destination)
        progress(1.0, desc="Done")
        yield str(destination), f"Completed: {len(chapters)} song folder(s) created for {safe_name(title)}."


def process_safe(
    url: str,
    media_file: str | None,
    tracklist: str,
    auto_split: bool,
    silence_threshold_db: float,
    min_silence_duration: float,
    min_track_duration: float,
    output_format: str,
):
    try:
        yield from process_stream(
            url,
            media_file,
            tracklist,
            auto_split,
            silence_threshold_db,
            min_silence_duration,
            min_track_duration,
            output_format,
        )
    except subprocess.CalledProcessError as error:
        details = error.stderr.strip() if error.stderr else str(error)
        yield None, f"Command failed: {details}"
    except Exception as error:
        yield None, str(error)


with gr.Blocks(title="Song Splitter") as demo:
    gr.Markdown("# Song Splitter\nExtract audio and split chaptered media into song-title folders.")
    with gr.Row():
        url = gr.Textbox(label="YouTube URL", placeholder=DEFAULT_URL)
        media_file = gr.File(label="Or choose a local audio/video file", type="filepath", file_types=[".mp4", ".mkv", ".webm", ".m4a", ".mp3", ".wav"])
    tracklist = gr.Textbox(
        label="Optional tracklist timestamps",
        placeholder="0:00 Intro\n3:12 Track Name\n...",
        lines=6,
    )

    with gr.Accordion("Auto split (advanced)", open=False):
        auto_split = gr.Checkbox(
            value=True,
            label="Auto split by silence when no chapters/tracklist are found",
        )

        silence_threshold_db = gr.Slider(
            minimum=-60,
            maximum=-10,
            value=-35,
            step=1,
            label="Silence threshold (dB)",
        )
        min_silence_duration = gr.Slider(
            minimum=0.1,
            maximum=3.0,
            value=0.8,
            step=0.1,
            label="Min silence duration (s)",
        )
        min_track_duration = gr.Slider(
            minimum=5,
            maximum=300,
            value=30,
            step=1,
            label="Min track duration (s)",
        )

    output_format = gr.Radio(["m4a", "mp3", "wav"], value="mp3", label="Output format")
    run_button = gr.Button("Extract and split", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)
    download = gr.File(label="Download ZIP", interactive=False)
    run_button.click(
        process_safe,
        inputs=[
            url,
            media_file,
            tracklist,
            auto_split,
            silence_threshold_db,
            min_silence_duration,
            min_track_duration,
            output_format,
        ],
        outputs=[download, status],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
