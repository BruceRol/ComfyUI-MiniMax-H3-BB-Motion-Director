# Portions derived from ComfyUI_MiniMaxH3_Director
# Copyright AIMixer and contributors
# Originally licensed under Apache License 2.0
# Modified for MiniMax H3 Motion Director, 2026-08-09
# This derivative project is distributed under GPL-3.0.
# See NOTICE and LICENSES/Apache-2.0-AIMixer.txt.

"""Shared helpers for the MiniMax H3 Motion Director timeline node."""

from __future__ import annotations

import json
import logging

from ..director.execution_report import append_report_section_lines
from ..director.plan import build_director_plan, count_all_timeline_segments, count_timeline_segments, plan_summary
from ..director.progress import report_director_planning
from ..lib.task_prompts import resolve_task_key, task_type_combo_options

log = logging.getLogger("ComfyUI-MiniMax-H3-Motion-Director")


def timeline_required_inputs() -> dict:
    """Timeline + prompt widgets shared by Director nodes."""
    combo_options, combo_meta = task_type_combo_options()
    return {
        "task_type": (combo_options, combo_meta),
        "global_prompt": (
            "STRING",
            {
                "default": "",
                "multiline": True,
                "tooltip": "Synced from in-node UI (global mode).",
            },
        ),
        "bd_grp_sample": ("BDGROUP", {"default": "采样设置"}),
        "cfg": (
            "FLOAT",
            {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "CFG for KSampler."},
        ),
        "seed": (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 0xFFFFFFFFFFFFFFFF,
                "control_after_generate": True,
                "tooltip": "Random seed for sampling.",
            },
        ),
        "frame_rate": (
            "FLOAT",
            {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01, "tooltip": "Timeline / output FPS (H3 trained at 24)."},
        ),
        "width": ("INT", {"default": 864, "min": 32, "max": 8192, "step": 32}),
        "height": ("INT", {"default": 480, "min": 32, "max": 8192, "step": 32}),
        "ref_max_size": ("INT", {"default": 864, "min": 32, "max": 8192, "step": 32}),
        "total_frames": (
            "INT",
            {
                "default": 124,
                "min": 5,
                "max": 100000,
                "tooltip": "Timeline total frames (fl2v = sum of shots). Per-shot generation still capped near 512.",
            },
        ),
        "timeline_data": (
            "STRING",
            {"default": "", "multiline": True, "tooltip": "Internal — video, segments, refs (populated by UI)."},
        ),
    }


def director_perf_inputs() -> dict:
    """Performance widgets shared by Director nodes."""
    return {
        "bd_grp_perf": ("BDGROUP", {"default": "性能"}),
        "clear_vram_between_segments": (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": "段间清理显存：每段结束后卸载模型并清空 CUDA 缓存。",
            },
        ),
        "low_ram": (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": (
                    "Low RAM retention. ON writes each clip as soon as it is final and "
                    "keeps only the head/tail continuity windows in RAM. OFF keeps every "
                    "exported chunk in RAM until the merge. Motion Context behaviour is "
                    "identical in both modes and no cache files are written."
                ),
            },
        ),
    }


def director_output_inputs() -> dict:
    """Per-clip h264 MP4 output widgets shared by Director nodes."""
    return {
        "bd_grp_output": ("BDGROUP", {"default": "输出"}),
        "dirname_prefix": (
            "STRING",
            {
                "default": "MiniMaxH3_BB",
                "multiline": False,
                "tooltip": "输出子目录名（ComfyUI output 目录下），每个片段写入 clip-XXX.mp4（从 1 开始编号，与编辑器一致）。",
            },
        ),
        "pix_fmt": (
            ["yuv420p", "yuv422p", "yuv444p"],
            {
                "default": "yuv420p",
                "tooltip": "h264 像素格式。yuv420p 兼容性最好。",
            },
        ),
        "crf": (
            "INT",
            {
                "default": 23,
                "min": 0,
                "max": 51,
                "tooltip": "h264 质量（CRF）。越低画质越高、文件越大。",
            },
        ),
    }


def default_timeline_json(
    *,
    task_type: str,
    global_prompt: str,
    total_frames: int,
    frame_rate: float,
    width: int,
    height: int,
    ref_max_size: int,
) -> str:
    return json.dumps(
        {
            "version": 4,
            "editMode": "global",
            "totalFrames": total_frames,
            "frameRate": frame_rate,
            "width": width,
            "height": height,
            "refMaxSize": ref_max_size,
            "output": {
                "mode": "fixed",
                "longEdge": ref_max_size,
                "width": width,
                "height": height,
                "maxExportFrames": 0,
                "exportMode": "all",
                "audioMode": "generate",
            },
            "videoClips": [],
            "video": {
                "fileName": "",
                "videoFile": "",
                "subfolder": "",
                "type": "input",
                "frames": [],
                "frameMap": [],
            },
            "global": {"taskType": task_type, "prompt": global_prompt, "refs": [], "referenceVideo": {}, "continuousReference": False},
            "segments": [
                {
                    "id": "s0",
                    "start": 0,
                    "length": total_frames,
                    "prompt": "",
                    "taskType": "",
                    "refs": [],
                    "referenceVideo": {},
                }
            ],
        },
        ensure_ascii=False,
    )


def _default_mixed_timeline_json(
    *,
    global_prompt: str,
    frame_rate: float,
    width: int,
    height: int,
    ref_max_size: int,
) -> str:
    """Create a valid root Mixed project when the user switches mode before editing cards."""
    return json.dumps(
        {
            "version": 1,
            "timelineMode": "mixed",
            "frameRate": frame_rate,
            "width": width,
            "height": height,
            "refMaxSize": ref_max_size,
            "output": {
                "mode": "fixed",
                "longEdge": ref_max_size,
                "width": width,
                "height": height,
                "maxExportFrames": 0,
                "exportMode": "all",
                "audioMode": "generate",
            },
            "runSelectEnabled": False,
            "runSelection": [],
            "segments": [
                {
                    "id": "seg_1",
                    "mode": "t2v",
                    "prompt": global_prompt or "",
                    "duration": 5.0,
                    "inputs": {"resultRefs": [], "identityPictures": []},
                    "continuity": {"visual": False, "audio": False},
                }
            ],
        },
        ensure_ascii=False,
    )


def prepare_director_plan(
    *,
    timeline_data: str,
    task_type: str,
    global_prompt: str,
    total_frames: int,
    frame_rate: float,
    width: int,
    height: int,
    ref_max_size: int,
    unique_id: str | None,
    motion_context_enabled: bool = True,
):
    from ..director.mixed_plan import build_mixed_director_plan, is_mixed_timeline

    task_key_requested = resolve_task_key(task_type)
    if not timeline_data or not timeline_data.strip():
        if task_key_requested == "mixed":
            timeline_data = _default_mixed_timeline_json(
                global_prompt=global_prompt,
                frame_rate=frame_rate,
                width=width,
                height=height,
                ref_max_size=ref_max_size,
            )
        else:
            timeline_data = default_timeline_json(
                task_type=task_type,
                global_prompt=global_prompt,
                total_frames=total_frames,
                frame_rate=frame_rate,
                width=width,
                height=height,
                ref_max_size=ref_max_size,
            )

    try:
        timeline_obj = json.loads(timeline_data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid timeline_data JSON: {exc}") from exc

    if is_mixed_timeline(timeline_obj, task_key_requested):
        plan = build_mixed_director_plan(
            timeline_obj,
            global_task_type=task_type,
            global_prompt=global_prompt,
            total_frames=total_frames,
            frame_rate=frame_rate,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
        )
        runnable = len(plan.run_indices) if plan.run_indices is not None else len(plan.segments)
        report_director_planning(
            unique_id,
            runnable,
            timeline_segment_total=len(plan.segments),
        )
        log.info("MiniMax H3 Motion Director: Mixed | %s", plan_summary(plan).replace("\n", " | "))
        return plan

    report_director_planning(
        unique_id,
        count_timeline_segments(timeline_data),
        timeline_segment_total=count_all_timeline_segments(timeline_data),
    )

    plan = build_director_plan(
        timeline_data,
        global_task_type=task_type,
        global_prompt=global_prompt,
        total_frames=total_frames,
        frame_rate=frame_rate,
        width=width,
        height=height,
        ref_max_size=ref_max_size,
        motion_context_enabled=motion_context_enabled,
    )
    log.info(plan_summary(plan).replace("\n", " | "))
    return plan


def finalize_director_report(plan, report: str, *, clip_count: int, extra_lines=None) -> str:
    """Report-only finalize: per-clip MP4 files replace the image/audio outputs."""
    if plan.export_mode == "segments":
        report = report + f"\n\nExport mode: segments — {clip_count} clip(s) written."
    else:
        report = report + f"\n\nExport mode: all — {clip_count} clip(s) written."
    if plan.run_indices is not None:
        report = report + f"\n\nPartial run: {clip_count} clip(s) written."
    if extra_lines:
        report = append_report_section_lines(report, "Timing", list(extra_lines))
    return report

