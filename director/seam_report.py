"""User-visible, non-destructive segment seam diagnostics."""

from __future__ import annotations

from typing import Any

import torch

from .segment_boundary import seam_diagnostics


def build_seam_report_lines(
    chunks: list[torch.Tensor] | tuple[torch.Tensor, ...],
    audios: list[dict[str, Any] | None] | tuple[dict[str, Any] | None, ...] | None,
    *,
    fps: float,
    segment_slots: list[int] | tuple[int, ...] | None = None,
    segment_lengths: list[int] | tuple[int, ...] | None = None,
) -> list[str]:
    """Format every exported boundary without changing frames or audio.

    ``segment_lengths`` reports the true clip lengths.  A streaming run passes the retained
    boundary views, whose ``shape[0]`` is the window size rather than the clip length, so
    the reported lengths must come from the segments themselves.
    """
    frames=list(chunks or [])
    audio_items=list(audios or [])
    slots=list(segment_slots if segment_slots is not None else range(len(frames)))
    lengths=[int(value) for value in (segment_lengths if segment_lengths is not None else [int(f.shape[0]) for f in frames])]
    if len(slots) != len(frames):
        raise ValueError("Seam report segment slot count must match exported chunks.")
    if len(lengths) != len(frames):
        raise ValueError("Seam report segment length count must match exported chunks.")
    if len(frames) < 2:
        return ["Status: NOT_APPLICABLE (single exported segment)"]
    lines=[]
    for i in range(len(frames)-1):
        left_audio=audio_items[i] if i < len(audio_items) else None
        right_audio=audio_items[i+1] if i+1 < len(audio_items) else None
        left_label=f"S{int(slots[i])+1}"
        right_label=f"S{int(slots[i+1])+1}"
        try:
            diag=seam_diagnostics(
                frames[i], frames[i+1], left_audio=left_audio, right_audio=right_audio, fps=float(fps)
            )
        except Exception as exc:
            lines.append(f"{left_label} -> {right_label}: UNAVAILABLE\n  Reason: {type(exc).__name__}: {exc}")
            continue
        shared_sr=int(diag.get("audio_sample_rate") or 0)
        left_sr=int(diag.get("left_audio_sample_rate") or 0)
        right_sr=int(diag.get("right_audio_sample_rate") or 0)
        if shared_sr > 0:
            audio_rate=f"{shared_sr} Hz"
        elif left_sr or right_sr:
            audio_rate=f"mismatch ({left_sr} -> {right_sr} Hz)"
        else:
            audio_rate="none"
        lines.append(
            f"{left_label} -> {right_label}: SUCCESS\n"
            f"  Exported Frames: {lengths[i]} -> {lengths[i+1]}\n"
            f"  Mean Abs RGB Jump: {float(diag['mean_abs_rgb_jump']):.6f}\n"
            f"  Luma Jump: {float(diag['luma_jump']):.6f}\n"
            f"  Audio Sample Rate: {audio_rate}\n"
            f"  Audio Samples: {int(diag['left_audio_samples'])} -> {int(diag['right_audio_samples'])}"
        )
    return lines


__all__=["build_seam_report_lines"]
