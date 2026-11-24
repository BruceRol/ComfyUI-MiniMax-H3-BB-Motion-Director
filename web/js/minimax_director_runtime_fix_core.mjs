export const GENERATED_AUDIO_CONTINUITY_TASKS = Object.freeze([
    "t2v",
    "i2v",
    "r2v",
    "fl2v",
    "v2v",
    "rv2v",
]);

const GENERATED_AUDIO_CONTINUITY_SET = new Set(
    GENERATED_AUDIO_CONTINUITY_TASKS,
);

export function generatedAudioContinuationShouldBeInteractive(
    taskKey,
    groupCount,
) {
    const task = String(taskKey || "")
        .trim()
        .toLowerCase();

    const count = Math.max(
        0,
        Math.trunc(Number(groupCount) || 0),
    );

    return (
        count > 1
        && GENERATED_AUDIO_CONTINUITY_SET.has(task)
    );
}

function assignOrDelete(target, key, value) {
    if (value === undefined) delete target[key];
    else target[key] = value;
}

export function restoreContinuityWidgetRenderer(widget) {
    const snapshot = widget?._mmxContinuityRendererSnapshot;
    if (!widget || !snapshot) return false;

    let changed = false;
    for (const key of ["type", "draw", "mouse", "onClick", "onPointerDown"]) {
        const before = widget[key];
        const after = snapshot[key];
        if (!Object.is(before, after)) changed = true;
        assignOrDelete(widget, key, after);
    }
    return changed;
}
