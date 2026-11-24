// Portions derived from ComfyUI_MiniMaxH3_Director
// Copyright AIMixer and contributors
// Originally licensed under Apache License 2.0
// Modified for MiniMax H3 Motion Director, 2026-08-09
// This derivative project is distributed under GPL-3.0.
// See NOTICE and LICENSES/Apache-2.0-AIMixer.txt.

// Shared timeline helpers for the BB Director frontend.
// The Inputs/Assets nodes were removed, so the socket/media helpers that used
// to live here are gone; only the timeline/task-key helpers remain.

export function resolveDirectorTaskKey(value) {
    const text = String(value || "").trim().toLowerCase();
    const match = text.match(/(?:^|\b)(mixed|rv2v|fl2v|r2v|i2v|v2v|t2v)(?:\b|$)/i);
    return match ? match[1].toLowerCase() : "t2v";
}

export function parseDirectorTimeline(raw) {
    if (raw && typeof raw === "object") return raw;
    const text = String(raw || "").trim();
    if (!text) return {};
    try {
        const parsed = JSON.parse(text);
        return parsed && typeof parsed === "object" && !Array.isArray(parsed)
            ? parsed
            : {};
    } catch {
        return {};
    }
}

export function directorGroupCount(timeline, mode) {
    const data = parseDirectorTimeline(timeline);
    const task = resolveDirectorTaskKey(mode);
    if (task === "fl2v" && Array.isArray(data.shots) && data.shots.length) {
        return Math.max(1, data.shots.length);
    }
    if (Array.isArray(data.segments) && data.segments.length) {
        return Math.max(1, data.segments.length);
    }
    return 1;
}
