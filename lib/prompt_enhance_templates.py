# Portions derived from ComfyUI_MiniMaxH3_Director
# Copyright AIMixer and contributors
# Originally licensed under Apache License 2.0
# Modified for MiniMax H3 Motion Director, 2026-08-09
# This derivative project is distributed under GPL-3.0.
# See NOTICE and LICENSES/Apache-2.0-AIMixer.txt.

"""MiniMax H3 per-task LLM enhancement templates (English only)."""

from __future__ import annotations

import re

from .official_pe_templates import (
    ENHANCE_TEMPLATES,
    JSON_MODE_TASKS,
    T2I_A14B_EN_SYS_PROMPT,
    T2I_SYSTEM_TASKS,
    T2V_A14B_EN_SYS_PROMPT,
    T2V_SYSTEM_TASKS,
)
from .task_prompts import get_task_prompt_spec

CHARACTER_DETAIL_NORMAL = "Normal"
CHARACTER_DETAIL_DETAILED = "Detailed"

DETAILED_MIN_TOTAL_CHARS = 300
DETAILED_MIN_APPEARANCE_CHARS = 200

_DEFAULT_USER_TEMPLATE = (
    "You are a helpful assistant that enhances prompts for video generation and editing. "
    "Rewrite the following instruction to be more detailed and specific. "
    "English only.\n\nInstruction: {user_prompt}"
)

_TEMPLATE_ALIASES = {"mv2v": "v2v", "vrc2v": "rv2v"}
_TASKS_REQUIRE_IMAGE_SLOTS = frozenset({"rv2v", "r2v", "r2i", "vi2v"})


def _task_key(task_type: str) -> str:
    return _TEMPLATE_ALIASES.get(task_type, task_type)


def uses_json_mode(task_type: str) -> bool:
    return _task_key(task_type) in JSON_MODE_TASKS


def uses_t2v_system_prompt(task_type: str) -> bool:
    return _task_key(task_type) in T2V_SYSTEM_TASKS


def uses_t2i_system_prompt(task_type: str) -> bool:
    return _task_key(task_type) in T2I_SYSTEM_TASKS


def _append_image_slot_rules(template: str, task_key: str) -> str:
    key = _task_key(task_key)
    if key not in _TASKS_REQUIRE_IMAGE_SLOTS:
        return template
    rule = (
        "\n\nCRITICAL (MiniMax H3 reference slots): every reference to a reference image "
        'MUST use "image0", "image1", … (lowercase, upload order). '
        'Example: "Replace the man with the woman from image0, preserving pose…". '
        'Do NOT use only "reference image" without the imageN tag. '
        "image0 is the first reference image, not a source video frame."
    )
    if rule.strip() not in template:
        template += rule
    return template


def normalize_character_detail_level(value: str) -> str:
    """Return 'detailed' or 'normal'."""
    v = (value or "").strip().lower()
    if v in ("detailed", "verbose", "full"):
        return "detailed"
    return "normal"


def is_detailed_character_level(level: str) -> bool:
    return normalize_character_detail_level(level) == "detailed"


def normalize_character_feature_enhance(value) -> bool:
    """Parse checkbox / bool / legacy combo string."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    v = str(value).strip().lower()
    if v in ("true", "1", "yes", "on", "detailed", "verbose", "full"):
        return True
    if v in ("false", "0", "no", "off", "normal", ""):
        return False
    return is_detailed_character_level(v)


def is_character_feature_enhance_enabled(
    feature_enhance=None,
    *,
    character_detail_level: str | None = None,
) -> bool:
    if feature_enhance is not None:
        return normalize_character_feature_enhance(feature_enhance)
    if character_detail_level is not None:
        return is_detailed_character_level(character_detail_level)
    return False


def count_chars(text: str) -> int:
    """Non-whitespace characters, used for the "too short" retry check."""
    return len(re.sub(r"\s+", "", text or ""))


def build_character_detail_directive(
    enabled,
    *,
    task_key: str = "",
    character_detail_level: str | None = None,
) -> str:
    """Extra LLM instruction when the character feature enhance option is checked."""
    if not is_character_feature_enhance_enabled(
        enabled,
        character_detail_level=character_detail_level,
    ):
        return ""
    key = _task_key(task_key)
    if key not in _TASKS_REQUIRE_IMAGE_SLOTS:
        return ""
    return (
        "[Character feature enhance — REQUIRED]\n"
        "Write rewritten_text in three parts, with the imageN character appearance as the core:\n"
        "(1) one edit line that mentions imageN;\n"
        f"(2) a long appearance paragraph (at least {DETAILED_MIN_APPEARANCE_CHARS} characters on its "
        "own, the bulk of the text) listing only features actually visible in the reference image:\n"
        "  - head: short/long/tied hair, hair colour, headwear (only when visible);\n"
        "  - face: face shape, skin tone, eyes/brows/lips (no makeup unless visible);\n"
        "  - top: the actual garment (lab coat / shirt / T-shirt / robe), colour, collar and sleeve shape;\n"
        "  - bottom and body: visible parts, belt or jewellery only when present;\n"
        "  - body type and setting: modern indoor / historical / outdoor.\n"
        "First decide whether the reference image is modern or historical and describe exactly what is "
        "there; never default to historical costume, never write source-video frame features into imageN, "
        "and never invent poetic wording without visual evidence.\n"
        "(3) one short preservation line.\n"
        f"Total length must be at least {DETAILED_MIN_TOTAL_CHARS} characters, and part 2 must be "
        "verifiable line by line against the reference image.\n\n"
    )


build_character_feature_enhance_directive = build_character_detail_directive


def build_detailed_retry_suffix(*, current_chars: int) -> str:
    return (
        f"\n\n[Too short — rewrite] Current output ~{current_chars} characters, below the character "
        f"feature enhance requirement. Output the JSON again with rewritten_text of at least "
        f"{DETAILED_MIN_TOTAL_CHARS} characters, where the appearance description for imageN takes at "
        f"least {DETAILED_MIN_APPEARANCE_CHARS} characters on its own and lists hair, headwear, facial "
        "features, makeup, top garment, bottom garment, belt, jewellery, body type and fabric texture. "
        "Do not summarize in one or two sentences."
    )


_RV2V_STATIC_INTRO_RE = re.compile(
    r"1\. The first 3 images are uniformly sampled frames from the \*\*source video\*\*.*?"
    r"3\. An original editing instruction \(the user's own wording\)\.\s*\n",
    re.DOTALL,
)


def patch_rv2v_vision_intro(
    template: str,
    *,
    source_count: int,
    ref_slots: list[int],
    ref_images_first: bool = False,
) -> str:
    """Replace the official static "first 3 images = source video" intro with the actual order."""
    if source_count <= 0 and not ref_slots:
        return template
    if not _RV2V_STATIC_INTRO_RE.search(template):
        return template

    ref_slots = list(ref_slots or [])
    ref_count = len(ref_slots)
    lines: list[str] = [
        "Vision attachment order (MUST follow — ignore any other attachment-order text below):"
    ]
    pos = 1
    if ref_images_first:
        for slot in ref_slots:
            lines.append(
                f"{pos}. Reference image image{slot} (appearance for 'from image{slot}' ONLY)."
            )
            pos += 1
        for i in range(source_count):
            lines.append(
                f"{pos}. Source video frame{i} (subject to replace — 'in the video' clause ONLY)."
            )
            pos += 1
    else:
        for i in range(source_count):
            lines.append(
                f"{pos}. Source video frame{i} (subject to replace — 'in the video' clause ONLY)."
            )
            pos += 1
        for slot in ref_slots:
            lines.append(
                f"{pos}. Reference image image{slot} (appearance for 'from image{slot}' ONLY)."
            )
            pos += 1
    lines.append("Original editing instruction:")

    intro = "\n".join(lines) + "\n\n"
    return _RV2V_STATIC_INTRO_RE.sub(intro, template, count=1)


def _raw_template(task_key: str) -> str:
    key = _task_key(task_key)
    return ENHANCE_TEMPLATES.get(key, _DEFAULT_USER_TEMPLATE)


def resolve_enhance_template(task_type: str, *, custom_template: str = "") -> str:
    """Return the user-message template (official by default)."""
    raw = (custom_template or "").strip()
    key = _task_key(task_type)
    if not raw:
        if key in T2V_SYSTEM_TASKS | T2I_SYSTEM_TASKS:
            return "{user_prompt}"
        raw = _raw_template(key)
        return _append_image_slot_rules(raw, key)
    return _append_image_slot_rules(raw, key)


def resolve_enhance_system_prompt(
    task_type: str,
    *,
    custom_template: str = "",
) -> str:
    """Official routing: t2v/t2i use A14B system prompts; others use task system prompts."""
    key = _task_key(task_type)
    if (custom_template or "").strip():
        return get_task_prompt_spec(key).system_prompt
    if key in T2V_SYSTEM_TASKS:
        return T2V_A14B_EN_SYS_PROMPT
    if key in T2I_SYSTEM_TASKS:
        return T2I_A14B_EN_SYS_PROMPT
    return get_task_prompt_spec(key).system_prompt


def format_enhance_user_content(
    template: str,
    *,
    user_prompt: str,
    image_num: int = 1,
) -> str:
    prompt = (user_prompt or "").strip()
    try:
        return template.format(
            user_prompt=prompt,
            original_text=prompt,
            image_num=image_num,
        )
    except KeyError:
        return template.format(user_prompt=prompt, image_num=image_num)


def get_enhance_template(task_type: str) -> str:
    return resolve_enhance_template(task_type)
