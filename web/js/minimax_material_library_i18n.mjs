// MiniMax H3 Motion Director — Material Library UI strings (English only).

const STRINGS = {
    button: "Library", title: "Material Library", close: "Close", cancel: "Close", apply: "Apply", applying: "Applying…",
    image: "Images", audio: "Audio", video: "Video", prompt: "Prompt", add: "Add material", search: "Search titles…",
    allCategories: "All", empty: "No materials yet", emptySearch: "No matching materials",
    selected: "Selected", preview: "Allocation preview", previewEmpty: "Nothing selected", target: "Apply to", common: "Common",
    targetRequired: "Choose an apply target first", promptCommonBlocked: "Prompts can only be applied to a Segment, not Common References.",
    append: "Append", replace: "Replace", promptMode: "Prompt mode", first: "First frame", last: "Last frame", fl2vRole: "Image role",
    leftRightHint: "Left-click adds · right-click removes the last occurrence", edit: "Edit", delete: "Delete", rename: "Title",
    category: "Category", content: "Prompt content", save: "Save", confirmDelete: "Delete “{title}”?",
    clearPage: "Clear current tab", clearAll: "Clear all selections", addCategory: "Add subcategory", renameCategory: "Rename subcategory", categoryName: "Subcategory name", zoom: "Zoom preview",
    upload: "Add from computer", uploading: "Saving {name}…", saved: "Saved", applied: "Materials applied to Director",
    error: "Error: {message}", sourceLocalOnly: "RV2V Source Video stays local-upload only; the current RV2V library offers Images, Audio and Prompt only.",
    promptNoShot: "FL2V cannot create an anchorless shot from Prompt alone.",
    v2vNoVideo: "V2V cannot create runnable segments from Prompt without Source Video.",
    noFreeSlot: "{type} reached the current target limit; later items were skipped.",
    createSegment: "Will create S{n}", itemTo: "{type} {order} → {target}", promptJoined: "Multiple Prompts are joined in queue order before applying.",
    firstFrame: "First Frame", lastFrame: "Last Frame", sourceVideo: "Source Video", refPicture: "Reference Picture",
    refAudio: "Reference Audio", refVideo: "Reference Video", picture: "Picture", segment: "S{n}",
    categories_image: ["Character", "Scene", "Prop", "Other"],
    categories_audio: ["Voice", "Dialogue", "SFX", "Music", "Other"],
    categories_video: ["Character", "Scene", "Motion", "Camera", "Other"],
    categories_prompt: ["Character", "Scene", "Motion", "Camera", "Style", "Dialogue", "Other"],
};

export function mlT(key, vars = {}) {
    const value = STRINGS[key] ?? key;
    if (Array.isArray(value)) return [...value];
    return String(value).replace(/\{(\w+)\}/g, (_, name) => String(vars[name] ?? ""));
}

export function materialCategories(kind) {
    return mlT(`categories_${kind}`);
}
