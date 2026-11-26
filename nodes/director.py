# Portions derived from ComfyUI_MiniMaxH3_Director
# Copyright AIMixer and contributors
# Originally licensed under Apache License 2.0
# Modified for MiniMax H3 Motion Director, 2026-08-09
# This derivative project is distributed under GPL-3.0.
# See NOTICE and LICENSES/Apache-2.0-AIMixer.txt.

"""MiniMax H3 Motion Director — timeline UI + official MiniMax H3 AV execution."""

from __future__ import annotations

import time

from ..director.execution_report import fmt_seconds
from ..director.executor_core import execute_director_plan_core
from ..director.progress import (
    report_director_audio_preview,
    report_director_complete,
    report_director_report,
)
from .director_common import (
    director_output_inputs,
    director_perf_inputs,
    finalize_director_report,
    prepare_director_plan,
    timeline_required_inputs,
)

_CATEGORY = "MiniMaxH3"

_DEFAULT_GLOBAL_PROMPT = "A cinematic scene with natural motion and synchronized ambience"


def director_timeline_required_inputs() -> dict:
    """Timeline widgets — defaults aligned with official MiniMax H3 workflow templates."""
    inputs = timeline_required_inputs()
    combo_options, combo_meta = inputs["task_type"]

    gp_meta = dict(inputs["global_prompt"][1])
    gp_meta["default"] = _DEFAULT_GLOBAL_PROMPT
    gp_meta["tooltip"] = (
        "User prompt — sent directly to MiniMaxH3ImageToVideo / ReferenceToVideo. "
        "r2v: <Picture 1>. v2v: source-timeline edit (<Video 1>). "
        "rv2v: source timeline + reference images (<Video 1> + <Picture N>). "
        "mixed: each segment compiles to an existing H3 task."
    )

    frames_meta = dict(inputs["total_frames"][1])
    frames_meta["default"] = 124
    frames_meta["tooltip"] = (
        "Frame count at 24 fps; snapped to MiniMax 17k+5 grid (124 ≈ 5s)."
    )

    return {
        **inputs,
        "task_type": (combo_options, combo_meta),
        "global_prompt": ("STRING", gp_meta),
        "total_frames": ("INT", frames_meta),
    }


class BB_MiniMaxH3MotionDirector:
    """In-node timeline Director using ComfyUI official MiniMax H3 pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "MiniMax H3 UNET (UNETLoader)."},
                ),
                "video_vae": (
                    "VAE",
                    {"tooltip": "MiniMax H3 video VAE (minimax_h3_video_vae)."},
                ),
                "audio_vae": (
                    "VAE",
                    {"tooltip": "MiniMax H3 audio VAE (minimax_h3_audio_vae). Required for r2v / v2v / rv2v."},
                ),
                "clip": (
                    "CLIP",
                    {"tooltip": "CLIPLoader type=minimax (qwen3vl)."},
                ),
                "sampler": (
                    "SAMPLER",
                    {
                        "tooltip": (
                            "Connect KSamplerSelect or MiniMax-H3 Turbo Sampler. "
                            "Internal sampling was removed: this connection is required."
                        )
                    },
                ),
                "sigmas": (
                    "SIGMAS",
                    {
                        "tooltip": (
                            "Connect BasicScheduler or another standard ComfyUI "
                            "scheduler built from the same H3 MODEL. Executed steps "
                            "come from the schedule length."
                        )
                    },
                ),
                **director_timeline_required_inputs(),
            },
            "optional": {
                "bd_grp_motion": ("BDGROUP", {"default": "Motion Context"}),
                "motion_context_enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Use the previous segment's final exported frames as "
                            "real H3 motion history. For segment N>1 this replaces "
                            "the old single-last-frame continuity handoff."
                        ),
                    },
                ),
                "context_length": (
                    "INT",
                    {
                        "default": 22,
                        "min": 1,
                        "max": 39,
                        "step": 1,
                        "tooltip": (
                            "Previous exported frames carried into the next segment. "
                            "Video mode uses H3-valid runs 1, 5, 22 or 39 and snaps "
                            "down when a shorter value is entered."
                        ),
                    },
                ),
                "source_overlap_frames": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 5,
                        "step": 5,
                        "tooltip": (
                            "V2V/RV2V only. 5 runs an independent H3-native bridge "
                            "across each eligible boundary: five original source frames "
                            "are conditioning only, generated frames at B-2/B+2 are "
                            "anchors, and regenerated B-1/B/B+1 replace the hard cut. "
                            "Visual Motion Context is skipped. 0 disables Source Bridge. "
                            "Mixed v1 always forces this to 0 because its Source Videos are segment-local."
                        ),
                    },
                ),
                "audio_context_enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Continue the previous segment's final exported generated "
                            "audio. Automatically disabled for source or mute audio mode."
                        ),
                    },
                ),
                "color_reanchor_enabled": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Re-anchor incoming Motion Context color statistics to a "
                            "stable reference to reduce cumulative color drift."
                        ),
                    },
                ),
                "bd_grp_advanced": ("BDGROUP", {"default": "高级采样"}),
                "refine_model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "Optional refine model. When empty the refine pass follows "
                            "the first-pass model. Must be a MiniMax H3 model carrying "
                            "ModelSamplingAV so the refine schedule can be validated."
                        )
                    },
                ),
                "refine_sampler": (
                    "SAMPLER",
                    {
                        "tooltip": (
                            "Optional refine sampler. Falls back to the shared "
                            "sampler connection when not connected."
                        )
                    },
                ),
                "refine_sigmas": (
                    "SIGMAS",
                    {
                        "tooltip": (
                            "Optional refine schedule. Falls back to the shared sigmas "
                            "connection when not connected. Executed refine steps = "
                            "schedule length x denoise."
                        )
                    },
                ),
                **director_perf_inputs(),
                "bd_grp_experimental": ("BDGROUP", {"default": "Experimental"}),
                "pin_renorm_enabled": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Re-normalize only the previous segment's "
                            "video latent tail to the first handoff std of the current "
                            "visual chain. Does not modify audio, source video, Picture "
                            "references, or RGB Color Re-anchor."
                        ),
                    },
                ),
                "postprocess_config": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Internal Director post-processing and Output Preview configuration.",
                    },
                ),
                **director_output_inputs(),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, **_kwargs):
        if input_types is not None:
            expected = {
                "model": "MODEL",
                "video_vae": "VAE",
                "audio_vae": "VAE",
                "clip": "CLIP",
                "sampler": "SAMPLER",
                "sigmas": "SIGMAS",
                "refine_model": "MODEL",
                "refine_sampler": "SAMPLER",
                "refine_sigmas": "SIGMAS",
            }
            for name, want in expected.items():
                got = input_types.get(name)
                if got is not None and got != want:
                    return f"{name}: expected {want}, linked node returns {got}."
            refine_connected = [
                name for name in ("refine_model", "refine_sampler", "refine_sigmas")
                if input_types.get(name) is not None
            ]
            if refine_connected and len(refine_connected) != 3:
                return (
                    "Refine sampling must be connected as a complete trio "
                    "(refine_model / refine_sampler / refine_sigmas); connected: "
                    + ", ".join(refine_connected)
                    + "."
                )
        return True

    OUTPUT_NODE = True
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_IS_LIST = ()
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "MiniMax H3 Motion Director: exported video/audio Motion Context, "
        "MiniMaxH3ImageToVideo / ReferenceToVideo conditioning, external "
        "SAMPLER+SIGMAS (required), and LTXVSeparateAVLatent decode. "
        "Supports t2v / i2v / fl2v / r2v / v2v / rv2v plus Mixed meta-mode. "
        "Media is referenced by filename from the timeline JSON. "
        "Writes one h264 MP4 per clip (with audio) under the output folder. "
        "Defaults: 0.4MP 16:9 (864×480), 5s / 124 frames @ 24 fps."
    )

    def execute(
        self,
        model,
        video_vae,
        audio_vae,
        clip,
        task_type,
        global_prompt,
        frame_rate,
        width,
        height,
        ref_max_size,
        total_frames,
        timeline_data,
        unique_id=None,
        motion_context_enabled=True,
        context_length=22,
        source_overlap_frames=5,
        audio_context_enabled=True,
        color_reanchor_enabled=False,
        sampler=None,
        sigmas=None,
        cfg=1.0,
        seed=0,
        refine_model=None,
        refine_sampler=None,
        refine_sigmas=None,
        clear_vram_between_segments=True,
        low_ram=True,
        pin_renorm_enabled=False,
        postprocess_config="",
        dirname_prefix="",
        pix_fmt="yuv420p",
        crf=23,
        prompt=None,
        extra_pnginfo=None,
        **kwargs,
    ):
        del kwargs, prompt, extra_pnginfo
        run_started = time.perf_counter()

        plan = prepare_director_plan(
            timeline_data=timeline_data,
            task_type=task_type,
            global_prompt=global_prompt,
            total_frames=total_frames,
            frame_rate=frame_rate,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
            unique_id=unique_id,
            motion_context_enabled=motion_context_enabled,
        )

        if bool(getattr(plan, "mixed_mode", False)):
            source_overlap_frames = 0
            # Mixed boundary toggles are per-link requests. The visible node-level
            # Motion/Audio Context switches remain the user-controlled global masters;
            # executor_core combines master AND ContextLink for the actual handoff.

        summary, segment_outputs, segment_audios, report = execute_director_plan_core(
            plan,
            node_id=unique_id,
            model=model,
            vae=video_vae,
            audio_vae=audio_vae,
            clip=clip,
            cfg=cfg,
            seed=seed,
            external_sampler=sampler,
            external_sigmas=sigmas,
            refine_model=refine_model,
            refine_sampler=refine_sampler,
            refine_sigmas=refine_sigmas,
            motion_context_enabled=motion_context_enabled,
            context_length=context_length,
            source_overlap_frames=source_overlap_frames,
            audio_context_enabled=audio_context_enabled,
            color_reanchor_enabled=color_reanchor_enabled,
            pin_renorm_enabled=pin_renorm_enabled,
            clear_vram_between_segments=clear_vram_between_segments,
            low_ram=low_ram,
            postprocess_config=postprocess_config,
            dirname_prefix=dirname_prefix,
            pix_fmt=pix_fmt,
            crf=crf,
        )

        progress_total = max(
            1,
            len(plan.run_indices) if plan.run_indices is not None else len(plan.segments),
        )
        report = finalize_director_report(
            plan,
            report,
            clip_count=int(getattr(summary, "clip_count", len(segment_outputs))),
            extra_lines=[f"End-to-end Total: {fmt_seconds(time.perf_counter() - run_started)}"],
        )
        print(report)

        report_director_report(unique_id, report)
        report_director_audio_preview(unique_id, segment_audios)
        report_director_complete(unique_id, progress_total)
        return {"ui": {"report": [report]}, "result": ()}
