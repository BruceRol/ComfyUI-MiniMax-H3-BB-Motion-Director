# Portions derived from ComfyUI_MiniMaxH3_Director
# Copyright AIMixer and contributors
# Originally licensed under Apache License 2.0
# Modified for MiniMax H3 Motion Director, 2026-08-09
# This derivative project is distributed under GPL-3.0.
# See NOTICE and LICENSES/Apache-2.0-AIMixer.txt.

"""Retained frame window for one clip in the streaming executor.

Continuity pair ops and Source Bridge writes are anchored at opposite ends of a
boundary: left-side ops read and write the previous chunk's last frames, right-side
ops read and write the next chunk's first frames.  A window keeps only the frames
those ops can reach, so the rest of the clip can be written to disk and released.

Window sizes must cover each op's *decision region*, not only its write region:
``_break_hold_pop_window(from_end=True)`` scans the last ``min(scan, n - 2)`` frames,
so the tail must be at least ``scan + 1`` frames.  Sizes come from ``director/low_ram.py``,
and the tail is widened by the configured Motion Context length when that pipeline is
active, so Low RAM ON and OFF produce the same context.

A chunk shorter than ``head + tail`` keeps the whole chunk, so short clips are always
faithful regardless of window sizes.

Writing order follows the ops.  The right-side ops run on ``head`` before it is written,
the left-side ops run on ``tail`` at the next boundary.  ``write_ready()`` therefore
flushes ``head + middle`` and holds ``tail``; ``finalize()`` flushes what is left.
"""

from __future__ import annotations

from typing import Optional

import torch

from .clip_writer import ClipWriter, clip_filename, silent_audio
from .low_ram import window_sizes
from .plan import SegmentPlan


class ClipWindow:
    """Head / middle / tail retention view of one segment chunk."""

    def __init__(
        self,
        seg: SegmentPlan,
        chunk: torch.Tensor,
        writer: ClipWriter,
        *,
        low_ram: bool = True,
        audio_dict: Optional[dict] = None,
        context_pipeline_active: bool = False,
        requested_context: int = 22,
    ) -> None:
        self.seg = seg
        self.writer = writer
        self.low_ram = bool(low_ram)
        self.context_pipeline_active = bool(context_pipeline_active)
        self.requested_context = int(requested_context)
        self.audio_dict = audio_dict or {}
        self.closed = False
        self._written_frames = 0
        self._prefix_written = False
        self._written_video = False

        n = int(chunk.shape[0])
        self.chunk_len = n
        head_len, tail_len = window_sizes(
            n,
            requested_context=self.requested_context,
            context_pipeline_active=self.context_pipeline_active,
        )
        self.head_len = head_len
        self.tail_len = tail_len
        self.middle_len = max(0, n - head_len - tail_len)

        if not self.low_ram or self.middle_len == 0:
            # Short chunk or full retention: the chunk is the authoritative storage and
            # both views alias regions of it.
            self.whole = chunk
            self.chunk = chunk
        else:
            # Only the head and the middle are writable before the next boundary; the
            # head is a clone so the chunk storage can be released once it has been
            # written, together with the middle.
            self.whole = None
            self.chunk = None

        self.head = chunk[:head_len]
        self.middle = chunk[head_len : head_len + self.middle_len]
        self.tail = chunk[n - tail_len :] if tail_len > 0 else chunk[:0]
        if self.whole is None:
            self.head = self.head.clone()
            self.tail = self.tail.clone()

        # Current views the continuity ops work on.  Ops may return clones, so the
        # caller must re-assign them with ``set_left_view`` / ``set_right_view``.
        self.left_view = self.tail
        self.right_view = self.head

    # --- absolute accessors -------------------------------------------------

    def frame_abs(self, abs_index: int) -> Optional[torch.Tensor]:
        """Frame at an absolute chunk index, or None when it was already released."""
        n = int(self.chunk_len)
        i = int(abs_index)
        if i < 0 or i >= n:
            return None
        if self.chunk is not None:
            return self.chunk[i]
        if i < self.head_len:
            return self.head[i]
        if i >= n - self.tail_len:
            return self.tail[i - (n - self.tail_len)]
        return None

    def set_frame_abs(self, abs_index: int, value: torch.Tensor) -> None:
        """Write one frame at an absolute chunk index; raises when it is not retained."""
        n = int(self.chunk_len)
        i = int(abs_index)
        if i < 0 or i >= n:
            raise IndexError(f"ClipWindow: frame {i} is outside segment #{self.seg.index}")
        if self.chunk is not None:
            self.chunk[i] = value
            return
        if i < self.head_len:
            self.head[i] = value
            return
        if i >= n - self.tail_len:
            self.tail[i - (n - self.tail_len)] = value
            return
        raise IndexError(
            f"ClipWindow: frame {i} of segment #{self.seg.index} was already written "
            "and is not retained."
        )

    # --- views ---------------------------------------------------------------

    def set_left_view(self, view: torch.Tensor) -> None:
        """Re-assign the tail view after an op that may have cloned it."""
        if self.chunk is not None:
            # The view is a region of the authoritative chunk: write the result back so
            # the other side of a short chunk sees the same pixels.
            start = int(self.chunk_len) - int(self.tail_len)
            length = int(view.shape[0])
            self.chunk[start : start + length] = view
            self.left_view = self.chunk[start : start + length]
            return
        self.left_view = view
        self.tail = view

    def set_right_view(self, view: torch.Tensor) -> None:
        """Re-assign the head view after an op that may have cloned it."""
        if self.chunk is not None:
            length = int(view.shape[0])
            self.chunk[:length] = view
            self.right_view = self.chunk[:length]
            return
        self.right_view = view
        self.head = view

    # --- writing -------------------------------------------------------------

    def write_ready(self) -> int:
        """Flush the frames no later boundary op can reach.

        Low RAM ON writes the head — already modified by the right-side ops — plus the
        untouched middle, and releases the chunk storage.  With full retention, or a chunk
        shorter than ``head + tail``, nothing can be released yet, so this is a no-op and
        ``finalize()`` writes the whole clip at the end.
        """
        if self.closed:
            raise RuntimeError(f"ClipWindow: segment #{self.seg.index} is already closed")
        if not self.low_ram or self.middle_len == 0:
            return 0
        parts = [self.right_view]
        if self.middle is not None and int(self.middle.shape[0]) > 0:
            parts.append(self.middle)
        frames = torch.cat(parts, dim=0)
        written = self.writer.write_frames(frames)
        self._written_frames += written
        self._prefix_written = True
        self.middle = None
        return written

    def finalize(self) -> int:
        """Write whatever is left of the clip, then its audio, and close the writer."""
        if self.closed:
            raise RuntimeError(f"ClipWindow: segment #{self.seg.index} is already closed")
        if self._written_video:
            pending = None
        elif self._prefix_written:
            pending = self.left_view
        elif self.chunk is not None:
            pending = self.chunk
        else:
            parts = [self.right_view]
            if self.middle is not None and int(self.middle.shape[0]) > 0:
                parts.append(self.middle)
            parts.append(self.left_view)
            pending = torch.cat(parts, dim=0)
        if pending is not None and int(pending.shape[0]) > 0:
            self._written_frames += self.writer.write_frames(pending)
        self._written_video = True
        self.write_audio(self.audio_dict)
        self.writer.close()
        self.closed = True
        return self._written_frames

    def write_audio(self, audio_dict: Optional[dict]) -> int:
        """Write the clip's audio track (silent padding when the segment has none)."""
        return self.writer.write_audio(audio_dict or silent_audio(self.chunk_len, fps=self.writer.fps, channels=self.writer.channels))

    def retained_frames(self) -> int:
        if self.chunk is not None:
            return int(self.chunk.shape[0])
        return int(self.head.shape[0]) + int(self.tail.shape[0])

    def retained_view(self) -> torch.Tensor:
        """The region the next boundary can still reach.

        In Low RAM this is the tail clone, so the caller can drop its reference to the
        full chunk and the storage is released once the head and middle have been written.
        With full retention, or a chunk shorter than the windows, it is the whole chunk.
        """
        if self.chunk is not None:
            return self.chunk
        return self.left_view

    def seam_view(self) -> torch.Tensor:
        """First and last exported frame, for the non-destructive seam report.

        The seam diagnostics read one frame per side (``left[-1]`` and ``right[0]``), so a
        two-frame view is enough; the reported lengths come from ``segment_lengths``.
        """
        parts = []
        right = self.right_view
        left = self.left_view
        if isinstance(right, torch.Tensor) and int(right.shape[0]) > 0:
            parts.append(right[:1])
        if isinstance(left, torch.Tensor) and int(left.shape[0]) > 0:
            parts.append(left[-1:])
        if not parts:
            return torch.zeros((0, 1, 1, 3), dtype=torch.float32)
        return torch.cat(parts, dim=0)

    # --- Motion Context ------------------------------------------------------

    def available_context_frames(self) -> int:
        """The segment's true length, exactly as ``select_context_span`` must see it."""
        return int(self.chunk_len)

    def context_frames(self, span: Optional[int] = None) -> Optional[torch.Tensor]:
        """Retained tail the next segment's Motion Context encodes.

        The tail window covers the configured ``context_length``, so this is the same
        region a full-retention run encodes.  ``None`` only after ``finalize()``.
        """
        frames = self.left_view
        if frames is None or int(frames.shape[0]) == 0:
            return None
        if span is None:
            return frames
        wanted = min(int(span), int(frames.shape[0]))
        return frames[-wanted:]

    def summary(self) -> dict:
        return {
            "segment_index": int(self.seg.index),
            "timeline_index": int(self.timeline_index),
            "low_ram": bool(self.low_ram),
            "context_pipeline_active": bool(self.context_pipeline_active),
            "requested_context": int(self.requested_context),
            "head": int(self.head_len),
            "middle": int(self.middle_len),
            "tail": int(self.tail_len),
            "retained": int(self.retained_frames()),
            "written_frames": int(self._written_frames),
            "closed": bool(self.closed),
        }

    @property
    def timeline_index(self) -> int:
        return int(self.seg.timeline_index)

    @property
    def filename(self) -> str:
        return clip_filename(self.timeline_index)

    @property
    def path(self) -> str:
        """Written clip path (empty until the writer has opened)."""
        return self.writer.path
