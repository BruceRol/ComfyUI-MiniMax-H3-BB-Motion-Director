# Portions derived from ComfyUI_MiniMaxH3_Director
# Copyright AIMixer and contributors
# Originally licensed under Apache License 2.0
# Modified for MiniMax H3 Motion Director, 2026-08-09
# This derivative project is distributed under GPL-3.0.
# See NOTICE and LICENSES/Apache-2.0-AIMixer.txt.

"""Collect vision inputs and media helpers for director prompt enhancement."""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import subprocess

import folder_paths
import numpy as np
import torch
from PIL import Image

from ..lib.prompt_enhance_templates import (
    DETAILED_MIN_APPEARANCE_CHARS,
    DETAILED_MIN_TOTAL_CHARS,
    normalize_character_feature_enhance,
)

log = logging.getLogger("ComfyUI-MiniMax-H3-Motion-Director.director")

# Prefer current MiniMax H3 names while retaining legacy fallbacks for existing setups.
LLM_VISION_MAX_SIDE = int(
    os.environ.get(
        "MINIMAX_H3_PE_VISION_MAX_SIDE",
        os.environ.get("BERNINI_PE_VISION_MAX_SIDE", "512"),
    )
)
LLM_VISION_JPEG_QUALITY = int(
    os.environ.get(
        "MINIMAX_H3_PE_VISION_JPEG_QUALITY",
        os.environ.get("BERNINI_PE_VISION_JPEG_QUALITY", "65"),
    )
)
LLM_VISION_MAX_IMAGES = int(
    os.environ.get(
        "MINIMAX_H3_PE_VISION_MAX_IMAGES",
        os.environ.get("BERNINI_PE_VISION_MAX_IMAGES", "4"),
    )
)


def downscale_b64_jpeg(
    b64: str,
    *,
    max_side: int = LLM_VISION_MAX_SIDE,
    quality: int = LLM_VISION_JPEG_QUALITY,
) -> str:
    """Shrink a base64 JPEG for LLM vision to reduce token / context usage."""
    if not b64:
        return b64
    try:
        raw = base64.b64decode(b64.split(",", 1)[-1] if b64.startswith("data:") else b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return b64.split(",", 1)[-1] if b64.startswith("data:") else b64


def prepare_llm_vision_images(
    images: list[str] | None,
    *,
    max_images: int = LLM_VISION_MAX_IMAGES,
    max_side: int = LLM_VISION_MAX_SIDE,
    quality: int = LLM_VISION_JPEG_QUALITY,
) -> list[str] | None:
    """Downscale and cap image count for local LLM vision (Ollama context limits)."""
    if not images:
        return None
    out = [
        downscale_b64_jpeg(img, max_side=max_side, quality=quality)
        for img in images
        if img
    ]
    if len(out) <= max_images:
        return out or None
    # Prefer: up to 2 source video frames + remaining reference slots.
    head = out[: min(2, max_images)]
    tail_n = max(0, max_images - len(head))
    trimmed = head + (out[-tail_n:] if tail_n else [])
    log.info(
        "LLM vision: trimmed %d -> %d images (max_side=%d) for context budget",
        len(out),
        len(trimmed),
        max_side,
    )
    return trimmed or None


def _tensor_frame_to_b64(frame: torch.Tensor, *, quality: int = 85) -> str | None:
    if frame is None or not isinstance(frame, torch.Tensor) or frame.numel() == 0:
        return None
    arr = frame.detach().float().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        return None
    arr = np.clip(arr[..., :3], 0.0, 1.0)
    img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def sample_video_tensor_frames(clip: torch.Tensor | None, *, max_frames: int = 3) -> list[str]:
    if clip is None or clip.shape[0] <= 0:
        return []
    total = int(clip.shape[0])
    if total <= max_frames:
        indices = list(range(total))
    else:
        step = total / (max_frames + 1)
        indices = [min(total - 1, max(0, int(step * (i + 1)))) for i in range(max_frames)]
    out: list[str] = []
    for idx in indices:
        b64 = _tensor_frame_to_b64(clip[idx : idx + 1])
        if b64:
            out.append(b64)
    return out


def parse_user_reference_slots(text: str) -> list[int]:
    """Extract MiniMax H3 image0–image4 indices from @imageN or imageN."""
    body = text or ""
    indices: list[int] = []
    for match in re.finditer(r"@image(\d)(?!\d)", body, re.IGNORECASE):
        index = int(match.group(1))
        if 0 <= index < 5 and index not in indices:
            indices.append(index)
    if indices:
        return sorted(indices)
    for match in re.finditer(r"(?<![@\w])image(\d)(?!\d)", body, re.IGNORECASE):
        index = int(match.group(1))
        if 0 <= index < 5 and index not in indices:
            indices.append(index)
    return sorted(indices)


def filter_vision_for_user_slots(
    images: list[str],
    source_count: int,
    ref_slots: list[int],
    user_slots: list[int],
) -> tuple[list[str], list[int], int]:
    """When user names imageN, keep source frames but send only those reference slots."""
    if not user_slots or not ref_slots or not images:
        return images, ref_slots, source_count
    user_set = set(user_slots)
    n_refs = len(ref_slots)
    sources = images[:source_count]
    refs = images[source_count : source_count + n_refs]
    tail = images[source_count + n_refs :]
    filtered_refs: list[str] = []
    filtered_slots: list[int] = []
    for slot, b64 in zip(ref_slots, refs):
        if slot in user_set:
            filtered_refs.append(b64)
            filtered_slots.append(slot)
    if not filtered_refs:
        return images, ref_slots, source_count
    log.info(
        "LLM vision: user requested image(s) %s — frames + refs %s",
        user_slots,
        filtered_slots,
    )
    return sources + filtered_refs + tail, filtered_slots, source_count


def is_replace_task_prompt(text: str) -> bool:
    """True when user prompt looks like reference-driven replacement."""
    return bool(re.search(r"\b(replace|swap|substitute|change)\b", text or "", re.IGNORECASE))


def reorder_vision_refs_first(
    images: list[str],
    source_count: int,
    ref_slots: list[int],
) -> tuple[list[str], int, list[int], bool]:
    """Put reference images before source frames so VL models anchor on imageN."""
    if source_count <= 0 or not ref_slots:
        return images, source_count, ref_slots, False
    sources = images[:source_count]
    n_refs = len(ref_slots)
    refs = images[source_count : source_count + n_refs]
    tail = images[source_count + n_refs :]
    return refs + sources + tail, source_count, ref_slots, True


def _has_image_tag(text: str, index: int) -> bool:
    """True when imageN appears in the text."""
    return bool(re.search(rf"(?<![@\w])image{index}(?!\d)", text or "", re.IGNORECASE))


def _normalize_reference_tags(text: str) -> str:
    """Normalize spacing and remove duplicated imageN phrases."""
    t = (text or "").strip()
    if not t:
        return t
    # "image1 from image1" -> "image1"
    t = re.sub(
        r"(image(\d))\s*(?:from|of)\s+\1\s*(?:the)?\s*",
        r"\1 ",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def ensure_user_reference_tags(text: str, user_slots: list[int]) -> str:
    """Re-inject imageN tags when the LLM drops them from the enhanced output."""
    if not text or not user_slots:
        return text
    result = _normalize_reference_tags(text)
    for index in user_slots:
        if _has_image_tag(result, index):
            continue
        tag = f"image{index}"
        injected = False
        for pattern, repl in (
            (r"(\breplace[sd]?\b[^,.;\n]{0,80}?\bwith\b)", rf"\1 {tag}"),
            (r"(\bsubstitute[sd]?\b[^,.;\n]{0,80}?\bwith\b)", rf"\1 {tag}"),
            (r"(\busing\b)", rf"\1 {tag}"),
        ):
            new_text, count = re.subn(pattern, repl, result, count=1, flags=re.IGNORECASE)
            if count:
                result = new_text
                injected = True
                break
        if not injected:
            result = f"Reference {tag}: {result}"
        log.info("Enhanced prompt missing %s — re-injected reference tag", tag)
    return _normalize_reference_tags(result)


def build_replace_source_target_directive(
    user_slots: list[int],
    *,
    source_count: int = 0,
) -> str:
    """Split the edit instruction: frame = subject to replace, imageN = replacement target."""
    if not user_slots or source_count <= 0:
        return ""
    slots_txt = ", ".join(f"image{s}" for s in user_slots)
    frames_txt = ", ".join(f"frame{i}" for i in range(source_count))
    return (
        "[Replace task — two sources, never swap them]\n"
        "The sentence must read: \"In the video, [current appearance of the subject to replace] is "
        "replaced by [target appearance] from imageN ...\"\n"
        f"1. Before 'replaced by': describe only what is actually visible in the source video frame "
        f"attachments ({frames_txt}) — the current clothing, hair and pose. Never write the reference "
        "image appearance in this clause.\n"
        f"2. After 'from {slots_txt}': describe only the appearance visible in the matching reference "
        "image attachment (the replacement target). Never write the source frame subject in this clause.\n"
        "Check each attachment before writing: when the frame subject and the reference target differ, "
        "describe each in its own clause; never mix, swap or invent them.\n\n"
    )


def build_replace_structured_json_directive(
    ref_slots: list[int],
    *,
    character_feature_enhance: bool = False,
) -> str:
    """Force split JSON fields so Python assembles the replace clause without swapping them."""
    if not ref_slots:
        return ""
    feature_enhance = normalize_character_feature_enhance(character_feature_enhance)
    keys_lines: list[str] = []
    for slot in ref_slots:
        keys_lines.append(
            f'  "image{slot}_target": "ONLY the visible appearance from reference image{slot} attachment"'
        )
    keys_block = ",\n".join(keys_lines)
    target_hint = (
        f"long itemized description, at least {DETAILED_MIN_TOTAL_CHARS} characters"
        if feature_enhance
        else "short and accurate"
    )
    return (
        "[Replace — structured JSON only, NO rewritten_text]\n"
        "Look at every vision attachment first (frame* = source video, imageN = reference), then fill the JSON.\n"
        "Return exactly one JSON object with these fixed keys:\n"
        '  "frame_subject": "ONLY the visible appearance of the subject to replace in the source video frame attachments",\n'
        f"{keys_block} ({target_hint}; never describe the frame subject here),\n"
        '  "preservation": "one sentence on the camera, background, lighting and motion that stay unchanged",\n'
        '  "scene_detail": "optional scene and atmosphere of the target video (do not repeat frame_subject)"\n'
        "Rules: frame_subject must come from the frame attachments, imageN_target from the imageN "
        "attachment. When the two appearances differ, describe each one correctly; never swap them. "
        "Do not include a rewritten_text field.\n\n"
    )


def assemble_replace_rv2v_prompt(
    data: dict,
    ref_slots: list[int],
) -> str:
    """Build the final replace prompt from the structured JSON (guaranteed clause order)."""
    if not data or not ref_slots:
        return ""
    frame = str(data.get("frame_subject") or data.get("frame_appearance") or "").strip()
    preserve = str(data.get("preservation") or "").strip()
    scene = str(data.get("scene_detail") or data.get("scene") or "").strip()
    slot = ref_slots[0]
    target = str(
        data.get(f"image{slot}_target")
        or data.get("reference_target")
        or data.get("image0_target")
        or ""
    ).strip()
    if not frame or not target:
        return ""
    core = f"Replace {frame} in the video with {target} from image{slot}"
    if preserve:
        core += f", {preserve}"
    if scene:
        core += f". {scene}"
    return _normalize_reference_tags(core)


def build_vision_attachment_banner(
    *,
    source_count: int = 0,
    ref_slots: list[int] | None = None,
    ref_video_count: int = 0,
) -> str:
    """Short banner matching the actual vision array order: frames first, then refs (official)."""
    ref_slots = list(ref_slots or [])
    if source_count <= 0 and not ref_slots and ref_video_count <= 0:
        return ""
    lines: list[str] = ["=== Vision attachments (same order as the image array below) ==="]
    pos = 1
    for i in range(source_count):
        lines.append(f"#{pos} = frame{i} source video → frame_subject / 'in the video' clause")
        pos += 1
    for slot in ref_slots:
        lines.append(f"#{pos} = image{slot} reference → image{slot}_target / 'from image{slot}'")
        pos += 1
    for _ in range(ref_video_count):
        lines.append(f"#{pos} = reference video frame (no imageN number)")
        pos += 1
    return "\n".join(lines) + "\n\n"


def build_user_image_directive(
    user_slots: list[int],
    *,
    character_feature_enhance: bool = False,
) -> str:
    if not user_slots:
        return ""
    feature_enhance = normalize_character_feature_enhance(character_feature_enhance)
    if len(user_slots) == 1:
        s = user_slots[0]
        detail_hint = (
            f"Expand the appearance visible in reference image{s} at length (at least "
            f"{DETAILED_MIN_APPEARANCE_CHARS} characters on its own, {DETAILED_MIN_TOTAL_CHARS} "
            "characters in total), item by item, and only list features actually visible in that image."
            if feature_enhance
            else "The appearance must match the reference attachment; never mix subjects or invent details."
        )
        return (
            f"[User specified image{s} — look at the reference attachment before writing image{s}]\n"
            f"Keep image{s} in the output; after 'from image{s}' write ONLY the real appearance of "
            f"reference image{s}. {detail_hint}\n"
            f"Before writing image{s}, confirm from the reference: 1) short or long hair, "
            f"2) the actual top garment and colour (lab coat / shirt / robe), 3) modern or historical setting. "
            "Describe what is actually there; never default to historical costume.\n"
            f"Never write source-video frame features into image{s}, and never use wording without visual evidence.\n"
            f"'In the video ...' describes only the frame subject. Output image{s} without the @ prefix.\n\n"
        )
    joined = ", ".join(f"image{s}" for s in user_slots)
    return (
        f"[User specified {joined}]\n"
        "Keep every imageN number and describe each one using only its own reference image; "
        "never mix subjects or invent details.\n"
        "Output image0/image1…, without the @ prefix and without the word slot.\n\n"
    )


# Backward-compatible alias for internal imports
build_user_slot_directive = build_user_image_directive


def build_vision_slot_preamble(
    *,
    source_count: int = 0,
    ref_slots: list[int] | None = None,
    ref_video_count: int = 0,
    ref_images_first: bool = False,
) -> str:
    """Explain the vision attachment order against the MiniMax H3 image0–4 slots for the LLM."""
    ref_slots = list(ref_slots or [])
    if source_count <= 0 and not ref_slots and ref_video_count <= 0:
        return ""
    lines: list[str] = ["Vision attachment → index map (MUST follow in the output):"]
    pos = 1
    if ref_images_first:
        for slot in ref_slots:
            lines.append(
                f"- Attachment #{pos} = reference image image{slot} "
                f"(use ONLY for the 'from image{slot}' appearance, checked against hair/clothing/colour)"
            )
            pos += 1
        for i in range(source_count):
            lines.append(
                f"- Attachment #{pos} = source video frame{i} "
                f"(use ONLY for 'in the video [subject to replace]')"
            )
            pos += 1
    else:
        for i in range(source_count):
            lines.append(f"- Attachment #{pos} = source video frame{i} (NOT image0/image1)")
            pos += 1
        for slot in ref_slots:
            lines.append(
                f"- Attachment #{pos} = reference image image{slot} "
                f"(write exactly image{slot} in the prompt)"
            )
            pos += 1
    for _ in range(ref_video_count):
        lines.append(f"- Attachment #{pos} = reference video frame (not an imageN number)")
        pos += 1
    if ref_slots:
        lines.append(
            f"CRITICAL: use image{ref_slots[0]}"
            + (f", image{ref_slots[1]}" if len(ref_slots) > 1 else "")
            + " etc.; source frames are frame0/frame1 only — never image0."
        )
    return "\n".join(lines) + "\n\n"


def refs_tensors_to_b64_with_slots(refs) -> tuple[list[str], list[int]]:
    """Return base64 images and MiniMax H3 slot indices sorted by slot."""
    items: list[tuple[int, str]] = []
    for ref in refs or []:
        tensor = getattr(ref, "tensor", None)
        if tensor is None:
            continue
        slot = int(getattr(ref, "index", getattr(ref, "slot", len(items))))
        if tensor.ndim == 4 and tensor.shape[0] > 0:
            b64 = _tensor_frame_to_b64(tensor[0:1])
        else:
            b64 = _tensor_frame_to_b64(tensor)
        if b64:
            items.append((slot, b64))
    items.sort(key=lambda pair: pair[0])
    return [b64 for _, b64 in items], [slot for slot, _ in items]


def refs_tensors_to_b64(refs) -> list[str]:
    images, _ = refs_tensors_to_b64_with_slots(refs)
    return images


def collect_segment_vision_b64(
    *,
    source_clip: torch.Tensor | None,
    refs,
    reference_video: torch.Tensor | None = None,
    max_video_frames: int = 2,
) -> tuple[list[str], int, int, list[int]]:
    """Return (images, ref_count, source_count, ref_slot_indices)."""
    source_frames = sample_video_tensor_frames(source_clip, max_frames=max_video_frames)
    ref_images, ref_slots = refs_tensors_to_b64_with_slots(refs)
    ref_video_frames = sample_video_tensor_frames(reference_video, max_frames=max_video_frames)
    all_images = source_frames + ref_images + ref_video_frames
    return all_images, len(ref_images), len(source_frames), ref_slots


def extract_input_video_frames_b64(
    filename: str,
    *,
    subfolder: str = "",
    num_frames: int = 3,
) -> tuple[list[str], str | None]:
    """Extract uniformly sampled JPEG base64 frames from a file in ComfyUI input/."""
    if not filename:
        return [], "No filename"
    input_dir = folder_paths.get_input_directory()
    safe = os.path.normpath(str(filename)).replace("\\", "/")
    if safe.startswith("..") or os.path.isabs(safe):
        return [], "Invalid filename"
    video_path = os.path.join(input_dir, subfolder, safe) if subfolder else os.path.join(input_dir, safe)
    if not os.path.isfile(video_path):
        return [], f"File not found: {filename}"

    num_frames = max(1, min(int(num_frames), 5))
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames,nb_frames",
                "-of",
                "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        total_str = (probe.stdout or "").strip().split(",")[0].strip()
        total_frames = int(total_str) if total_str.isdigit() else 100
        if total_frames <= num_frames:
            indices = list(range(total_frames))
        else:
            step = total_frames / (num_frames + 1)
            indices = [int(step * (i + 1)) for i in range(num_frames)]

        frames_b64: list[str] = []
        for idx in indices:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                    "-vf",
                    f"select=eq(n\\,{idx})",
                    "-frames:v",
                    "1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "mjpeg",
                    "-q:v",
                    "4",
                    "pipe:1",
                ],
                capture_output=True,
                timeout=15,
            )
            if result.returncode == 0 and result.stdout:
                frames_b64.append(base64.b64encode(result.stdout).decode("ascii"))
        return frames_b64, None
    except FileNotFoundError:
        return [], "ffmpeg/ffprobe not found"
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def load_input_image_b64(filename: str) -> tuple[str | None, str | None]:
    if not filename:
        return None, "No filename"
    input_dir = folder_paths.get_input_directory()
    safe = os.path.normpath(str(filename)).replace("\\", "/")
    if safe.startswith("..") or os.path.isabs(safe):
        return None, "Invalid filename"
    path = os.path.join(input_dir, safe)
    if not os.path.isfile(path):
        return None, f"File not found: {filename}"
    try:
        img = Image.open(path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii"), None
    except Exception as exc:
        return None, str(exc)
