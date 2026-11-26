"""Runtime materialization of Mixed Segment Result still references.

Cross-run cache reuse is gone: a Mixed reference is resolved from the frames the
executor retained for it during the same run. ``expand_run_selection`` guarantees
the producer is always generated before its consumer, so the reference is either
already retained or the run is invalid.
"""

from __future__ import annotations

from typing import Any

import torch


_ROLE_INDEX = {
    "i2v_start": 0,
    "fl2v_first": 0,
    "fl2v_last": 1,
}


def select_result_frame(frames: torch.Tensor, selector: str | int) -> torch.Tensor:
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or int(frames.shape[0]) <= 0:
        raise ValueError("Mixed result reference source has no generated IMAGE frames.")
    if selector == "last":
        index = int(frames.shape[0]) - 1
    else:
        index = int(selector)
    if index < 0 or index >= int(frames.shape[0]):
        raise ValueError(
            f"Mixed result reference frame {index} is outside source result "
            f"0..{int(frames.shape[0]) - 1}."
        )
    return frames[index : index + 1]


def capture_mixed_reference_frames(plan, seg, frames: torch.Tensor) -> None:
    """Retain the stills later Mixed consumers reference while the producer is in memory.

    Called by the executor at segment capture time, when the full chunk is available.
    Under the Low-RAM window policy the producer chunk may already be released by the
    time a consumer samples, so the referenced frames are stored here instead of the
    whole segment.
    """
    if not bool(getattr(plan, "mixed_mode", False)):
        return
    store = getattr(plan, "mixed_reference_frames", None)
    if store is None:
        store = {}
        plan.mixed_reference_frames = store
    stable_id = str(getattr(seg, "stable_id", ""))
    for consumer in plan.segments:
        for descriptor in (getattr(consumer, "mixed_result_refs", None) or []):
            if str(descriptor.get("segmentId") or "").strip() != stable_id:
                continue
            frame = descriptor.get("frame", "last")
            bucket = store.setdefault(int(seg.index), {})
            key = str(frame)
            if key not in bucket:
                bucket[key] = select_result_frame(frames, frame).detach().cpu().float().contiguous()


class MixedResultRef:
    """SegmentRef-compatible object whose tensor resolves only when conditioning needs it."""

    def __init__(
        self,
        *,
        index: int,
        role: str,
        source_segment_id: str,
        frame: str | int,
        plan,
    ) -> None:
        self.index = int(index)
        self.role = str(role)
        self.source_segment_id = str(source_segment_id)
        self.frame = frame
        self.asset_id = f"mixed-result:{self.source_segment_id}:{self.frame}:{self.role}"
        self._plan = plan
        self._resolved: torch.Tensor | None = None

    def _source_segment(self):
        for segment in self._plan.segments:
            if str(getattr(segment, "stable_id", "")) == self.source_segment_id:
                return segment
        raise ValueError(
            f"Missing Reference: Mixed source segment {self.source_segment_id!r} no longer exists."
        )

    @property
    def tensor(self) -> torch.Tensor:
        if self._resolved is not None:
            return self._resolved

        producer = self._source_segment()
        store = getattr(self._plan, "mixed_reference_frames", None) or {}
        retained = store.get(int(producer.index), {}).get(str(self.frame))
        if retained is None:
            raise ValueError(
                f"Mixed dependency unavailable: Segment {getattr(producer, 'timeline_index', producer.index) + 1} "
                f"({self.source_segment_id}) must be generated before it can provide frame {self.frame!r}."
            )
        self._resolved = retained
        return self._resolved


def _source_id_for_ref(plan, consumer_index: int, ref: dict[str, Any]) -> str:
    # Legacy Previous/Earlier compatibility is deliberately handled only by
    # mixed_schema.normalize_mixed_segments(). Runtime receives one stable-ID
    # representation so reorder/delete/selective-run cannot diverge by origin.
    if str(ref.get("origin") or "") != "segment":
        raise ValueError("Invalid Mixed result reference: non-canonical origin reached runtime.")
    source_id = str(ref.get("segmentId") or "").strip()
    if not source_id:
        raise ValueError("Missing Reference: Segment Result has no stable segment id.")

    source_index = next(
        (
            index
            for index, segment in enumerate(plan.segments)
            if str(getattr(segment, "stable_id", "")) == source_id
        ),
        None,
    )
    if source_index is None:
        raise ValueError(f"Missing Reference: Mixed source segment {source_id!r} no longer exists.")
    if source_index >= int(consumer_index):
        raise ValueError(
            f"Invalid Reference: Mixed Segment Result {source_id!r} is not earlier than its consumer."
        )
    return source_id


def _next_identity_slot(refs) -> int:
    used = {int(getattr(ref, "index", -1)) for ref in refs or []}
    for index in range(9):
        if index not in used:
            return index
    raise ValueError("Mixed identity references exceed MiniMax H3 Picture limit (9).")


def attach_mixed_result_refs(plan) -> None:
    """Attach SegmentRef-compatible lazy stills to compiled Mixed segments."""
    if not bool(getattr(plan, "mixed_mode", False)):
        return

    for consumer_index, seg in enumerate(plan.segments):
        descriptors = list(getattr(seg, "mixed_result_refs", None) or [])
        if not descriptors:
            continue
        for descriptor in descriptors:
            role = str(descriptor.get("role") or "identity")
            source_id = _source_id_for_ref(plan, consumer_index, descriptor)
            frame = descriptor.get("frame", "last")

            if role == "identity":
                if str(getattr(seg, "mixed_mode", "")) not in {"r2v", "source_video"}:
                    raise ValueError(
                        f"Invalid Mixed identity reference on {getattr(seg, 'mixed_mode', '') or seg.task_key}."
                    )
                slot = _next_identity_slot(seg.refs)
            elif role in _ROLE_INDEX:
                slot = _ROLE_INDEX[role]
                expected = {
                    "i2v_start": "i2v",
                    "fl2v_first": "fl2v",
                    "fl2v_last": "fl2v",
                }[role]
                if str(getattr(seg, "mixed_mode", "")) != expected:
                    raise ValueError(f"Invalid Mixed {role} reference on {getattr(seg, 'mixed_mode', '')} segment.")
                seg.refs = [
                    ref
                    for ref in (seg.refs or [])
                    if int(getattr(ref, "index", -1)) != slot
                ]
                if role == "i2v_start":
                    seg.source_clip = None
            else:
                raise ValueError(f"Unsupported Mixed result reference role: {role}.")

            seg.refs.append(
                MixedResultRef(
                    index=slot,
                    role=role,
                    source_segment_id=source_id,
                    frame=frame,
                    plan=plan,
                )
            )
        seg.refs = sorted(seg.refs, key=lambda ref: int(getattr(ref, "index", 0)))


__all__ = [
    "MixedResultRef",
    "attach_mixed_result_refs",
    "capture_mixed_reference_frames",
    "select_result_frame",
]
