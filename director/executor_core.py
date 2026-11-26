"""Facade for the preserved MiniMax H3 executor.

The executor implementation is kept in ``executor_core_legacy.py``.  This facade runs it
with a per-call copy of its globals so the Advanced Sampling connections and the Motion
Context encode timings reach the preserved call sites without global monkeypatches.

The facade returns the ``ClipStreamState`` the legacy core built: the per-clip MP4 writer
is the output now, so there is no combined video tensor.
"""

from __future__ import annotations

import functools
import logging
import time
import types
from typing import Any

from . import executor_core_legacy as _legacy
from .core_sampling import resolve_refine_model, resolve_sampling_mode, validate_external_sampling
from .postprocess_config import normalize_postprocess_config
from .progress import current_director_progress_context
from .refine_sampling import apply_global_refine

log = logging.getLogger(__name__)


class _TimedVAE:
    """Delegate a VAE while measuring only encode calls made through the proxy."""

    def __init__(self, target, bucket: dict[str, float], key: str):
        self._target = target
        self._bucket = bucket
        self._key = key

    def __getattr__(self, name: str):
        return getattr(self._target, name)

    def encode(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return self._target.encode(*args, **kwargs)
        finally:
            self._bucket[self._key] = self._bucket.get(self._key, 0.0) + (
                time.perf_counter() - started
            )


def _sampling_context_partials(
    external_sampler,
    external_sigmas,
    refine_model=None,
    refine_sampler=None,
    refine_sigmas=None,
) -> dict[str, Any]:
    """Bind the Advanced Sampling connections for the refine stages.

    The preserved executor calls the refine stages without the external
    SAMPLER/SIGMAS objects, so the facade supplies them through its private
    globals copy instead of rewriting the legacy call sites. The optional
    refine trio overrides the shared connection when fully connected.
    """
    return {
        "apply_global_refine": functools.partial(
            apply_global_refine,
            external_sampler=external_sampler,
            external_sigmas=external_sigmas,
            refine_model=refine_model,
            refine_sampler=refine_sampler,
            refine_sigmas=refine_sigmas,
        ),
    }


def _legacy_runner(function_globals: dict[str, Any]) -> types.FunctionType:
    """The preserved executor bound to a private globals dictionary."""
    runner = types.FunctionType(
        _legacy.execute_director_plan_core.__code__,
        function_globals,
        name=_legacy.execute_director_plan_core.__name__,
        argdefs=_legacy.execute_director_plan_core.__defaults__,
        closure=_legacy.execute_director_plan_core.__closure__,
    )
    runner.__kwdefaults__ = dict(_legacy.execute_director_plan_core.__kwdefaults__ or {})
    return runner


def _append_executed_steps_report(report: str, executed_steps: int) -> str:
    """State the schedule actually executed.

    The preserved report prints the internal sampler/scheduler pair, which no
    longer exists, so the facade records the connected SIGMAS length instead.
    """
    return (report or "") + "\n[Executed Sampling]\nSteps: " + str(int(executed_steps)) + " (connected SIGMAS)"


def _append_context_encode_report(report: str, context_timings: dict[int, dict[str, float]]) -> str:
    lines = ["", "[Motion Context Encode Timing]"]
    if not context_timings:
        lines.append("No Motion Context encode calls in this run.")
    else:
        for slot, timing in sorted(context_timings.items()):
            lines.append(
                f"S{slot + 1}: calls={int(timing.get('calls', 0))}; "
                f"VideoVAE Encode={float(timing.get('video_vae', 0.0)):.2f}s; "
                f"AudioVAE Refresh={float(timing.get('audio_vae', 0.0)):.2f}s; "
                f"Context/Repin Total={float(timing.get('total', 0.0)):.2f}s"
            )
    return (report or "") + "\n" + "\n".join(lines)


def execute_director_plan_core(
    plan,
    *,
    node_id: str | None = None,
    model,
    vae,
    audio_vae,
    clip,
    cfg: float = 1.0,
    seed: int = 0,
    # steps / sampler / scheduler / shifts are accepted for the frozen legacy
    # contract but no longer select anything: internal sampling is gone, so the
    # executed schedule is always the connected SIGMAS and the executed step
    # count is the length of that schedule.
    steps: int = 25,
    sampler: str = "",
    scheduler: str = "",
    shift_video: float = 12.0,
    shift_audio: float = 3.0,
    external_sampler=None,
    external_sigmas=None,
    refine_model=None,
    refine_sampler=None,
    refine_sigmas=None,
    motion_context_enabled: bool = True,
    context_length: int = 22,
    source_overlap_frames: int = 5,
    audio_context_enabled: bool = True,
    color_reanchor_enabled: bool = False,
    pin_renorm_enabled: bool = False,
    clear_vram_between_segments: bool = True,
    # The canvas is not a facade argument: the legacy resolves the authoritative
    # stride-aligned canvas from the plan itself, and the writer uses that.
    postprocess_config: str | dict[str, Any] = "",
    dirname_prefix: str = "",
    pix_fmt: str = "yuv420p",
    crf: int = 23,
    low_ram: bool = True,
):
    postprocess = normalize_postprocess_config(postprocess_config)

    # Internal sampling is gone: the executed schedule is always the connected
    # SIGMAS, and the executed step count is the length of that schedule.
    sampling_mode = resolve_sampling_mode(external_sampler, external_sigmas)
    pass_model, _pass_model_label = resolve_refine_model(refine_model, model)
    pass_sampler = refine_sampler if refine_sampler is not None else external_sampler
    pass_sigmas = refine_sigmas if refine_sigmas is not None else external_sigmas
    _, executed_steps = validate_external_sampling(pass_model, pass_sampler, pass_sigmas)
    # The frozen legacy report reads this key; it now reports executed steps.
    postprocess["global_refine"]["steps"] = executed_steps

    call_kwargs = {
        "node_id": node_id,
        "model": model,
        "vae": vae,
        "audio_vae": audio_vae,
        "clip": clip,
        "cfg": cfg,
        "seed": seed,
        # Frozen legacy contract: the executed count is the connected SIGMAS
        # schedule length, not a widget value.
        "steps": executed_steps,
        "sampler": sampler,
        "scheduler": scheduler,
        "shift_video": shift_video,
        "shift_audio": shift_audio,
        "external_sampler": external_sampler,
        "external_sigmas": external_sigmas,
        "motion_context_enabled": motion_context_enabled,
        "context_length": context_length,
        "source_overlap_frames": source_overlap_frames,
        "audio_context_enabled": audio_context_enabled,
        "color_reanchor_enabled": color_reanchor_enabled,
        "pin_renorm_enabled": pin_renorm_enabled,
        "clear_vram_between_segments": clear_vram_between_segments,
        "postprocess_config": postprocess,
        "dirname_prefix": dirname_prefix,
        "pix_fmt": pix_fmt,
        "crf": int(crf),
        "low_ram": bool(low_ram),
    }

    context_timings: dict[int, dict[str, float]] = {}

    def context_hook(conditioning, **kwargs):
        active_slot = int(current_director_progress_context().get("timeline_segment") or 1) - 1
        timing = context_timings.setdefault(
            active_slot,
            {"calls": 0.0, "video_vae": 0.0, "audio_vae": 0.0, "total": 0.0},
        )
        local_timing: dict[str, float] = {}
        call = dict(kwargs)
        if call.get("video_vae") is not None:
            call["video_vae"] = _TimedVAE(call["video_vae"], local_timing, "video")
        if call.get("audio_vae") is not None:
            call["audio_vae"] = _TimedVAE(call["audio_vae"], local_timing, "audio")

        started = time.perf_counter()
        merged, info = _legacy.apply_exported_motion_context(conditioning, **call)
        elapsed = time.perf_counter() - started
        timing["calls"] += 1
        timing["video_vae"] += float(local_timing.get("video", 0.0))
        timing["audio_vae"] += float(local_timing.get("audio", 0.0))
        timing["total"] += elapsed
        return merged, info

    # Per-call copy: the hook updates below must not mutate the legacy module globals.
    # The runner receives them through _legacy_runner, never as a call argument.
    function_globals = dict(_legacy.execute_director_plan_core.__globals__)
    function_globals.update(_sampling_context_partials(
        external_sampler,
        external_sigmas,
        refine_model,
        refine_sampler,
        refine_sigmas,
    ))
    function_globals.update(
        apply_exported_motion_context=context_hook,
        report_director_report=lambda *_args, **_kwargs: None,
    )
    runner = _legacy_runner(function_globals)

    stream, segment_outputs, segment_audios, report = runner(plan, **call_kwargs)
    report = _append_executed_steps_report(report, executed_steps)
    report = _append_context_encode_report(report, context_timings)
    return stream, segment_outputs, segment_audios, report


def __getattr__(name: str):
    return getattr(_legacy, name)


__all__ = ["execute_director_plan_core"]
