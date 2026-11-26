"""Low-RAM retention policy for the BB director.

The legacy executor keeps every exported segment chunk in RAM until the final
assembly.  Only the continuity windows around each boundary are needed to finish
a clip, so in Low RAM mode a segment retains just its head and tail window and the
rest of it is written to disk as soon as it is final.

Low RAM OFF keeps the full chunks in RAM.  No segment cache files are written in
either mode: retention is a memory policy only.

Retaining a window only preserves the legacy decisions when every decision that reads a
window is made from the window plus the segment's *true* frame count.  Motion Context
therefore reports ``frame_count`` / ``export_frames`` from the full segment length while
reading its frames from the tail window, and Color Re-anchor builds its chain-root
baseline from the full chunk at capture time, before any window is written.
"""

from __future__ import annotations

from typing import Any

# Tail window: the widest decision region of any right-edge op.
# ``_break_hold_pop_window(from_end=True)`` scans ``min(scan, n - 2)`` pairs from the
# end, so with CONTINUITY_HOLD_POP_SCAN = 6 the earliest pair it can consider is frame
# ``n - 7``; the ``n - 2`` guard skips one more frame, so the window must be scan + 2 = 8
# frames to make the same decision as on a full chunk.  ``_micro_seam_bridge`` writes the
# last three frames, which fits inside that window.
TAIL_WINDOW_FRAMES = 8

# Head window: ``_additive_opening_luma`` grades the body opening over
# CONTINUITY_SEAM_ADD_LUMA_FRAMES (12) frames, the widest region any left-edge op reads
# and writes.  ``_ease_opening_spikes`` scans CONTINUITY_SEAM_SCAN_FRAMES (5) from index 0
# and ``_soften_body0_toward_prev`` writes index 0, both inside 12.
HEAD_WINDOW_FRAMES = 12

# Motion Context decision region.  ``select_context_span`` snaps the configured
# ``context_length`` widget down to the largest run of VIDEO_CONTEXT_GRID
# (39, 22, 5, 1) that fits the previous segment, and ``_encode_video_context`` encodes
# ``frames[-span:]``.  The tail must therefore cover the whole configured value, so the
# window is sized from the parameter and never from a fixed constant: Low RAM ON and OFF
# must produce the same Motion Context.  ``MAX_CONTEXT_FRAMES`` is the widget maximum and
# the largest H3-valid run, so a window sized from the parameter can never exceed it.
MAX_CONTEXT_FRAMES = 39


def resolve_low_ram(value: Any = None, *, default: bool = True) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("on", "true", "1", "yes"):
            return True
        if text in ("off", "false", "0", "no"):
            return False
    return bool(value)


def context_tail_frames(requested_context: int) -> int:
    """Tail frames one segment must retain for the next segment's Motion Context.

    Sized from the configured ``context_length``, not a constant: the retained tail is the
    exact region ``select_context_span`` / ``_encode_video_context`` read, so Low RAM ON
    and OFF give the same context.  The segment's *true* frame count is still reported as
    ``available`` to ``select_context_span``, so the span decision is unchanged.
    """
    return min(max(1, int(requested_context)), MAX_CONTEXT_FRAMES)


def tail_window_frames(
    *, requested_context: int = 22, context_pipeline_active: bool = False
) -> int:
    """Tail window for one chunk, widened by the configured context span when active."""
    tail = TAIL_WINDOW_FRAMES
    if context_pipeline_active:
        tail = max(tail, context_tail_frames(requested_context))
    return tail


def window_sizes(
    chunk_frames: int,
    *,
    requested_context: int = 22,
    context_pipeline_active: bool = False,
) -> tuple[int, int]:
    """Head and tail window sizes for one chunk (never larger than the chunk)."""
    n = int(chunk_frames)
    head = min(HEAD_WINDOW_FRAMES, n)
    tail = min(
        tail_window_frames(
            requested_context=requested_context,
            context_pipeline_active=context_pipeline_active,
        ),
        max(0, n - head),
    )
    return head, tail


__all__ = [
    "HEAD_WINDOW_FRAMES",
    "MAX_CONTEXT_FRAMES",
    "TAIL_WINDOW_FRAMES",
    "context_tail_frames",
    "resolve_low_ram",
    "tail_window_frames",
    "window_sizes",
]
