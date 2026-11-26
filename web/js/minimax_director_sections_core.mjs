export const DIRECTOR_SECTION_IDS = Object.freeze([
    "bd_grp_sample",
    "bd_grp_motion",
    "mmx_postprocess_group",
    "bd_grp_perf",
]);

const COPY = Object.freeze({
    sections: {
        bd_grp_sample: "Sampling Settings",
        bd_grp_motion: "Cross-Segment Continuity",
        mmx_postprocess_group: "Post Processing",
        bd_grp_perf: "Performance",
    },
    samplingStates: {
        external: "External",
        incomplete: "Incomplete",
    },
    widgets: {
        seed: "Seed",
        control_after_generate: "Control After Generate",
        "control after generate": "Control After Generate",
        motion_context_enabled: "Motion Context",
        context_length: "Context Frames",
        pin_renorm_enabled: "Latent Scale Lock",
        mmx_pin_renorm_proxy: "Latent Scale Lock",
        source_overlap_frames: "Source Video Bridge",
        audio_context_enabled: "Continue Generated Audio",
        color_reanchor_enabled: "Color Re-anchor",
        mmx_global_refine_proxy: "Global Refine",
        clear_vram_between_segments: "Clear VRAM Between Segments",
        low_ram: "Low RAM Retention",
    },
});

export function sectionTitle(sectionId) {
    return COPY.sections[sectionId] || String(sectionId || "");
}

export function samplingHeaderText(state) {
    const normalized = state === "external" ? "external" : "incomplete";
    return `${COPY.sections.bd_grp_sample}  [${COPY.samplingStates[normalized]}]`;
}

export function localizedWidgetLabel(name) {
    return COPY.widgets[String(name || "")] || String(name || "");
}
