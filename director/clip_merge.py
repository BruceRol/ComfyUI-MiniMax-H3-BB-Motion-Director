"""End-of-run clip merge for the BB director.

Only "Export all" merges: the per-clip MP4 files written during the run are concatenated
in export order into ``full_video.mp4``, in the same ``dirname_prefix`` subdirectory.
Nothing is read back from memory — the merge works purely from the files on disk.

Stream copy is safe because the writer guarantees identical parameters for every clip
(h264, the same pixel format, the same padded even canvas, the same fps, AAC at 44100 Hz
on every clip).  When a probe finds a mismatch the merge re-encodes instead of copying.

Clip files are removed only after a successful merge.  A single clip is renamed.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from typing import Optional

import folder_paths

from .clip_writer import OUTPUT_SAMPLE_RATE, PRESET

log = logging.getLogger(__name__)

MERGED_FILENAME = "full_video.mp4"


class MergeResult:
    def __init__(
        self,
        path: str,
        method: str,
        removed: list[str],
        lines: list[str],
    ) -> None:
        self.path = path
        self.method = method
        self.removed = list(removed)
        self.lines = list(lines)


def probe_clip(path: str) -> Optional[dict]:
    """Stream parameters that must match for a stream-copy concat."""
    from ..lib.video_io import ffprobe_bin

    probe = ffprobe_bin()
    if not probe:
        return None
    try:
        res = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,codec_name,pix_fmt,width,height,r_frame_rate,sample_rate",
                "-of",
                "json",
                path,
            ],
            capture_output=True,
            check=True,
        )
        payload = json.loads(res.stdout.decode("utf-8", "replace"))
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError) as exc:
        log.debug("clip merge probe failed for %s: %s", path, exc)
        return None
    info: dict = {}
    for stream in payload.get("streams") or []:
        kind = stream.get("codec_type")
        if kind == "video":
            info["video"] = (
                stream.get("codec_name"),
                stream.get("pix_fmt"),
                int(stream.get("width") or 0),
                int(stream.get("height") or 0),
                stream.get("r_frame_rate"),
            )
        elif kind == "audio":
            info["audio"] = (stream.get("codec_name"), int(stream.get("sample_rate") or 0))
    return info or None


def _concat_list_path(paths: list[str]) -> str:
    list_path = os.path.join(tempfile.gettempdir(), "bb_clip_concat.txt")
    with open(list_path, "w", encoding="utf-8") as handle:
        for path in paths:
            handle.write("file '%s'\n" % str(path).replace("'", "'\\''"))
    return list_path


def _run_ffmpeg(args: list[str]) -> tuple[bool, str]:
    try:
        res = subprocess.run(args, capture_output=True)
    except OSError as exc:
        return False, str(exc)
    if res.returncode == 0:
        return True, ""
    return False, res.stderr.decode("utf-8", "replace").strip()


def merge_clips(
    stream,
    *,
    crf: int = 23,
    pix_fmt: str = "yuv420p",
) -> Optional[MergeResult]:
    """Concatenate the written clips in export order; returns None when nothing was written."""
    paths = [path for path in stream.paths() if path and os.path.isfile(path)]
    if not paths:
        return None

    out_dir = os.path.join(folder_paths.get_output_directory(), stream.manager.dirname)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, MERGED_FILENAME)

    # One clip: the file already is the whole export, so rename it.
    if len(paths) == 1:
        try:
            os.replace(paths[0], out_path)
        except OSError as exc:
            line = f"Merged: skipped ({type(exc).__name__}: {exc}); clip file kept"
            log.warning("[BB Merge] %s", line)
            return MergeResult(paths[0], "skipped", [], [line])
        print(f"[BB Merge] {out_path}")
        return MergeResult(
            out_path,
            "rename",
            [],
            [f"Merged: {MERGED_FILENAME} (single clip, renamed from {os.path.basename(paths[0])})"],
        )

    from ..lib.audio_io import _ffmpeg_bin

    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        line = "Merged: skipped (ffmpeg not found); clip files kept"
        log.warning("[BB Merge] %s", line)
        return MergeResult(None, "skipped", [], [line])

    probes = [probe_clip(path) for path in paths]
    # A missing probe means ffprobe is unavailable; the writer guarantees identical
    # parameters for every clip, so stream copy is still the intended path.
    copy_safe = all(probe == probes[0] for probe in probes)

    list_path = _concat_list_path(paths)
    base = [ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", list_path]

    method = "copy" if copy_safe else "reencode"
    args = base + ["-c", "copy", "-movflags", "+faststart", out_path]
    ok, error = _run_ffmpeg(args)
    if not ok:
        # Re-encode fallback: mismatched parameters, or a copy the container rejected.
        method = "reencode"
        args = base + [
            "-c:v",
            "libx264",
            "-preset",
            PRESET,
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            pix_fmt if pix_fmt else "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            str(OUTPUT_SAMPLE_RATE),
            "-movflags",
            "+faststart",
            out_path,
        ]
        ok, error = _run_ffmpeg(args)
        if not ok:
            line = f"Merged: FAILED ({error or 'ffmpeg exited non-zero'}); clip files kept"
            log.warning("[BB Merge] %s", line)
            return MergeResult(None, "failed", [], [line])

    try:
        os.unlink(list_path)
    except OSError:
        pass

    removed = []
    for path in paths:
        try:
            os.unlink(path)
            removed.append(path)
            print(f"[BB Merge] removed {path}")
        except OSError as exc:
            log.warning("[BB Merge] could not remove %s: %s", path, exc)

    print(f"[BB Merge] {out_path}")
    lines = [
        f"Merged: {MERGED_FILENAME} ({len(paths)} clip(s), {method})",
        f"Removed clip files: {len(removed)}",
    ]
    return MergeResult(out_path, method, removed, lines)


__all__ = ["MERGED_FILENAME", "MergeResult", "merge_clips", "probe_clip"]
