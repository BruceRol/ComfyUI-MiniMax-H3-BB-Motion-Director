# Portions derived from ComfyUI_MiniMaxH3_Director
# Copyright AIMixer and contributors
# Originally licensed under Apache License 2.0
# Modified for MiniMax H3 Motion Director, 2026-08-09
# This derivative project is distributed under GPL-3.0.
# See NOTICE and LICENSES/Apache-2.0-AIMixer.txt.

"""MiniMax H3 task_type labels and combo options (freeform Qwen prompts — no T5 system prefix)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskPromptSpec:
    key: str
    label: str
    system_prompt: str
    description: str


TASK_PROMPT_SPECS: tuple[TaskPromptSpec, ...] = (
    TaskPromptSpec(
        "default",
        "Default (general)",
        "",
        "MiniMax H3 uses freeform Qwen3-VL prompts, no T5 system prefix.",
    ),
    TaskPromptSpec(
        "t2v",
        "Text to Video",
        "",
        "Text to video+audio; no first frame / reference images.",
    ),
    TaskPromptSpec(
        "i2v",
        "Image to Video",
        "",
        "First-frame image to video+audio (ImageToVideo + first_frame).",
    ),
    TaskPromptSpec(
        "fl2v",
        "First-Last Frame to Video",
        "",
        "First frame + last frame constraint (ImageToVideo + first_frame + last_frame).",
    ),
    TaskPromptSpec(
        "r2v",
        "Reference to Video",
        "",
        "Grouped references to video (like first/last-frame groups): each group takes "
        "Picture 1-9, Audio 1-3, Video 1-3; prompts use <Picture N> / <Video K> / <Audio J>. "
        "For source-video timeline editing use v2v/rv2v.",
    ),
    TaskPromptSpec(
        "v2v",
        "Video to Video",
        "",
        "Upload a source video and edit it segment by segment on the timeline; each segment's "
        "source frames go to ReferenceToVideo as <Video 1> (no reference image slots).",
    ),
    TaskPromptSpec(
        "rv2v",
        "Reference Video Edit",
        "",
        "Source-video timeline editing with optional reference images (Picture 1-9) and "
        "reference audio (Audio 1-3); each segment's source frames are <Video 1>, reference "
        "images are <Picture N>, reference audio is <Audio J>; without assets it behaves like v2v.",
    ),
    TaskPromptSpec(
        "mixed",
        "Mixed",
        "",
        "Per-segment choice of T2V / I2V / FL2V / R2V / Source Video on one timeline. "
        "Mixed is a Director meta-mode: each segment is compiled to an existing H3 backend task before execution.",
    ),
)

TASK_PROMPT_BY_KEY = {spec.key: spec for spec in TASK_PROMPT_SPECS}
HIDDEN_TASK_TYPE_KEYS: frozenset[str] = frozenset()


def task_type_option_label(spec: TaskPromptSpec) -> str:
    return f"{spec.key} — {spec.label}"


def task_type_combo_options() -> tuple[list[str], dict]:
    options = [
        task_type_option_label(spec)
        for spec in TASK_PROMPT_SPECS
        if spec.key not in HIDDEN_TASK_TYPE_KEYS and spec.key != "default"
    ]
    default_spec = TASK_PROMPT_BY_KEY["t2v"]
    return options, {
        "default": task_type_option_label(default_spec),
        "tooltip": (
            "MiniMax H3 Director supports t2v / i2v / fl2v / r2v / v2v / rv2v / mixed. "
            "Mixed is the per-segment meta-mode; its segments compile to existing H3 tasks and "
            "mixed is never sent to the model. Prompts go straight to MiniMaxH3ImageToVideo or "
            "MiniMaxH3ReferenceToVideo (tokenized internally)."
        ),
    }


def resolve_task_key(task_type_value: str) -> str:
    value = task_type_value.split(",[object Object]", 1)[0].strip()
    if " · " in value:
        value = value.split(" · ", 1)[0].strip()
    for sep in (" — ", " —— ", " - ", " – "):
        if sep in value:
            return value.split(sep, 1)[0].strip()
    return value


def get_task_prompt_spec(task_type_value: str) -> TaskPromptSpec:
    key = resolve_task_key(task_type_value)
    return TASK_PROMPT_BY_KEY.get(key, TASK_PROMPT_BY_KEY["default"])


def apply_task_system_prompt(task_type_value: str, positive_prompt: str) -> str:
    """H3 nodes tokenize raw user prompt — no system prefix injection."""
    del task_type_value
    return positive_prompt
