"""In-run Motion Context handoff store for the BB director.

Motion Context is a handoff between segments of the same run: what one segment
produces is kept in RAM and read by the next one.  Nothing is written to disk, so
there is no cache folder, no versioned payload and no stale-file validation.  A
segment that was not generated in this run has nothing to hand off.

The retained region is the window the consumer reads, sized from the configured
``context_length`` (see ``director/low_ram.py``) and never from a fixed constant, so
Low RAM ON and OFF produce exactly the same context.  The consumer still decides its
span from the segment's true frame count, so the span decision is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

LATENT_HANDOFF_PIPELINE = "motion_context_latent_tail_v6_color_chain_pin_composed"


@dataclass(frozen=True)
class RetainedMotionContext:
    frames: torch.Tensor | None
    audio: dict[str, Any] | None
    latent: dict[str, Any] | None = None
    handoff: dict[str, Any] | None = None


def _sample_streams(latent: dict[str, Any]) -> list[Any]:
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if hasattr(samples, "unbind"):
        return list(samples.unbind())
    if isinstance(samples, (tuple, list)):
        return list(samples)
    if isinstance(samples, torch.Tensor):
        return [samples]
    return []


def latent_context_canvas(latent: dict[str, Any] | None) -> tuple[int, int] | None:
    """Return the retained H3 video-latent canvas as pixel width/height."""
    if not isinstance(latent, dict):
        return None
    streams = _sample_streams(latent)
    if not streams:
        return None
    video = streams[0]
    if not isinstance(video, torch.Tensor) or video.ndim not in {4, 5}:
        return None
    return int(video.shape[-1]) * 16, int(video.shape[-2]) * 16


def prepare_latent_context_tail(
    latent: dict[str, Any],
    handoff: dict[str, Any],
    *,
    span: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep only the reusable H3 AV-latent endpoint window in RAM.

    ``span`` is the window the next segment is allowed to read, i.e. the configured
    Motion Context length.  The returned handoff uses a local 0..span timeline: the tail
    is self-contained, so no original coordinate is needed to consume it.
    """
    required = {"context_end_frame", "trim_frames", "export_frames", "sample_frames"}
    if not isinstance(handoff, dict) or not required.issubset(handoff):
        raise ValueError("AV latent handoff metadata is incomplete.")

    from .motion_context import (
        _audio_context_from_latent,
        select_context_span,
        video_context_from_latent,
    )

    retained_frames = select_context_span(span, int(handoff["export_frames"]))
    blocks, _offsets, selected_end = video_context_from_latent(
        latent,
        span=retained_frames,
        context_end_frame=int(handoff["context_end_frame"]),
    )
    video_tail = torch.cat(blocks, dim=2).detach().cpu().contiguous()
    streams: list[torch.Tensor] = [video_tail]
    if len(_sample_streams(latent)) >= 2:
        audio_ref, _audio_steps = _audio_context_from_latent(
            latent,
            span=retained_frames,
            context_end_frame=selected_end,
        )
        streams.append(audio_ref["audio_latent"].detach().cpu().contiguous())

    tail_latent: dict[str, Any] = {"samples": tuple(streams)}
    for key, value in latent.items():
        if key != "samples" and not isinstance(value, torch.Tensor):
            tail_latent[key] = value

    tail_handoff = {
        "context_end_frame": retained_frames,
        "trim_frames": 0,
        "export_frames": retained_frames,
        "sample_frames": retained_frames,
        "visual_latent_valid": bool(handoff.get("visual_latent_valid", True)),
    }
    baseline = handoff.get("pin_renorm_baseline_std")
    if baseline is not None:
        tail_handoff["pin_renorm_baseline_std"] = float(baseline)
    from .color_reanchor import validate_color_anchor_statistics

    color_anchor = validate_color_anchor_statistics(handoff.get("color_anchor_stats"))
    if color_anchor is not None:
        tail_handoff["color_anchor_stats"] = color_anchor
    return tail_latent, tail_handoff


__all__ = [
    "LATENT_HANDOFF_PIPELINE",
    "RetainedMotionContext",
    "latent_context_canvas",
    "prepare_latent_context_tail",
]
