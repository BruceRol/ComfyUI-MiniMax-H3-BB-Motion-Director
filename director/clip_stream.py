"""Per-clip streaming state for the MiniMax H3 BB director.

The legacy executor accumulated every exported segment chunk and then assembled them
into one combined tensor.  This module keeps the same chunk contents but writes each
clip to disk as soon as it is final, so no combined video is ever built.

A clip is final when its boundary with the next segment has been applied: the Source
Bridge write and the continuity pair ops run on the two retained windows, then the
previous clip is flushed and closed.  The current clip's head is written right after
that boundary (the pair ops may have modified it), its tail is held for the next one.

Window sizes and the decision-region rule are in ``director/low_ram.py``.  When the
motion context pipeline or the Source Bridge feature is active there are no pair ops,
so every chunk is final at capture and the windows are only a memory policy.  In that
case the tail window widens to the configured Motion Context length, so Low RAM ON and
OFF produce exactly the same context for the next segment.
"""

from __future__ import annotations

from typing import Optional

import torch

from .clip_window import ClipWindow
from .clip_writer import ClipWriterManager, audio_channels
from .low_ram import resolve_low_ram
from .plan import SegmentPlan


class ClipStreamState:
    """One open clip per segment, written incrementally in export order."""

    def __init__(
        self,
        plan,
        manager: ClipWriterManager,
        *,
        low_ram: Optional[bool] = None,
        requested_context: int = 22,
        run_indices=None,
    ) -> None:
        self.plan = plan
        self.manager = manager
        self.low_ram = resolve_low_ram(low_ram, default=True)
        self.requested_context = int(requested_context)
        # The segments this run actually generates.  ``plan.run_indices`` is the timeline
        # selection, which is not the export order in "Export all" mode, so the executor
        # passes its own resolved set.
        self.run_indices = set(int(index) for index in (run_indices or set()))
        self.windows: dict[int, ClipWindow] = {}

    def open_window(
        self,
        seg: SegmentPlan,
        chunk: torch.Tensor,
        audio_dict: Optional[dict] = None,
        *,
        context_pipeline_active: bool = False,
    ) -> ClipWindow:
        """Open the writer and the retention window for one generated chunk.

        ``context_pipeline_active`` is the executor's own flag: the tail window widens to
        the configured ``requested_context`` only when Motion Context is actually running.
        """
        index = int(seg.index)
        if index in self.windows:
            raise RuntimeError(f"ClipStreamState: segment #{index} is already open")
        writer = self.manager.writer(int(seg.timeline_index), channels=audio_channels(audio_dict))
        window = ClipWindow(
            seg,
            chunk,
            writer,
            low_ram=self.low_ram,
            audio_dict=audio_dict,
            context_pipeline_active=context_pipeline_active,
            requested_context=self.requested_context,
        )
        self.windows[index] = window
        return window

    def write_ready(self, window: ClipWindow) -> int:
        """Flush the frames no later boundary op can reach."""
        return window.write_ready()

    def finalize(self, window: ClipWindow) -> int:
        """Flush the retained boundary, write audio, close the clip."""
        return window.finalize()

    def finalize_open(self) -> list[ClipWindow]:
        """Close every clip that is still open, in export order."""
        for window in self.export_order():
            if not window.closed:
                self.finalize(window)
        return list(self.export_order())

    def export_order(self) -> list[ClipWindow]:
        """Windows in the order the clips are exported (compact run order)."""
        indices = sorted(self.windows)
        run = self.run_indices or set(
            int(index) for index in (getattr(self.plan, "run_indices", None) or set())
        )
        if run:
            indices = [index for index in indices if index in run]
        return [self.windows[index] for index in indices]

    def paths(self) -> list[str]:
        """Written clip paths, in export order."""
        return [window.path for window in self.export_order() if window.path]

    @property
    def clip_count(self) -> int:
        return len(self.export_order())

    def window(self, segment_index: int) -> Optional[ClipWindow]:
        """Window for one segment, or None when it was never opened."""
        return self.windows.get(int(segment_index))

    def report_lines(self) -> list[str]:
        """Writer report lines, in export order."""
        return self.manager.summary_lines()

    def summary(self) -> dict:
        return {
            "low_ram": bool(self.low_ram),
            "requested_context": int(self.requested_context),
            "clip_count": self.clip_count,
            "clips": [window.summary() for window in self.export_order()],
        }
