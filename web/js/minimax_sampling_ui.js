const SEED_CONTROL_MODES = new Set(["fixed", "increment", "decrement", "randomize"]);

export function normalizeSeedControlMode(value) {
    const mode = String(value || "").trim().toLowerCase();
    return SEED_CONTROL_MODES.has(mode) ? mode : "unknown";
}

export function seedControlModeFromWidgets(widgets = []) {
    const seed = widgets.find((widget) => widget?.name === "seed");
    const control = seed?.linkedWidgets?.find((widget) => (
        /control[_\s]?after[_\s]?generate/i.test(String(widget?.name || widget?.label || ""))
    ));
    return normalizeSeedControlMode(control?.value);
}

function hasLink(input) {
    return input?.link !== null && input?.link !== undefined;
}

// Sampling is external-only: the Director has no internal sampler, so both the
// SAMPLER and SIGMAS sockets must be connected for the node to run.
export function getSamplingConnectionState(inputs = []) {
    const sampler = inputs.find((input) => input?.name === "sampler");
    const sigmas = inputs.find((input) => input?.name === "sigmas");
    if (hasLink(sampler) && hasLink(sigmas)) return "external";
    return "incomplete";
}

const DIRECTOR_GROUP_INPUT_NAMES = new Set([
    "bd_grp_sample",
    "bd_grp_motion",
    "bd_grp_advanced",
    "bd_grp_perf",
    "bd_grp_experimental",
]);

// Current Director widget serialization order. BDBROUP widgets are real widgets
// when the frontend extension is loaded. If the extension fails during a graph
// load, ComfyUI can persist them as input sockets instead, shifting every later
// widget value by one slot. Repair that graph shape before node construction.
const DIRECTOR_GROUP_WIDGET_LAYOUT = Object.freeze([
    [2, "采样设置"],
    [12, "Motion Context"],
    [18, "高级采样"],
    [24, "性能"],
    [27, "Experimental"],
]);

function isDirectorWorkflowNode(node) {
    return String(node?.type || node?.comfyClass || "") === "BB_MiniMaxH3MotionDirector";
}

function removePersistedDirectorGroupInputs(node) {
    if (!Array.isArray(node?.inputs)) return [];
    const removed = [];
    node.inputs.forEach((input, index) => {
        if (DIRECTOR_GROUP_INPUT_NAMES.has(String(input?.name || ""))) removed.push(index);
    });
    if (!removed.length) return removed;
    const removeSet = new Set(removed);
    node.inputs = node.inputs.filter((_, index) => !removeSet.has(index));
    return removed;
}

function restoreMissingDirectorGroupWidgetValues(node) {
    const values = node?.widgets_values;
    if (!Array.isArray(values)) return 0;
    let inserted = 0;
    for (const [index, fallback] of DIRECTOR_GROUP_WIDGET_LAYOUT) {
        // Every BDBROUP value is a string; the widget immediately following each
        // group is numeric/boolean. This makes a missing group slot unambiguous.
        if (index <= values.length && typeof values[index] !== "string") {
            values.splice(index, 0, fallback);
            inserted += 1;
        }
    }
    return inserted;
}

export function repairDirectorGroupWidgetWorkflow(graphData) {
    if (!graphData || !Array.isArray(graphData.nodes)) return 0;
    const removedByNode = new Map();
    let repaired = 0;
    for (const node of graphData.nodes) {
        if (!isDirectorWorkflowNode(node)) continue;
        const removed = removePersistedDirectorGroupInputs(node);
        if (removed.length) {
            removedByNode.set(String(node.id), removed);
            repaired += removed.length;
        }
        repaired += restoreMissingDirectorGroupWidgetValues(node);
    }
    if (!removedByNode.size || !Array.isArray(graphData.links)) return repaired;

    const adjustedSlot = (nodeId, slot) => {
        const removed = removedByNode.get(String(nodeId));
        if (!removed?.length) return slot;
        const original = Number(slot);
        if (!Number.isFinite(original)) return slot;
        return original - removed.filter((index) => index < original).length;
    };

    for (const link of graphData.links) {
        if (Array.isArray(link)) {
            link[4] = adjustedSlot(link[3], link[4]);
            continue;
        }
        if (!link || typeof link !== "object") continue;
        const targetIdKey = "target_id" in link ? "target_id" : "targetId";
        const targetSlotKey = "target_slot" in link ? "target_slot" : "targetSlot";
        link[targetSlotKey] = adjustedSlot(link[targetIdKey], link[targetSlotKey]);
    }
    return repaired;
}

