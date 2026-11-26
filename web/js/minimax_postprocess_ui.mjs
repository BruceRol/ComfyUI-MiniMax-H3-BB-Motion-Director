const DEFAULT_CONFIG = Object.freeze({
    version: 9,
    global_refine: {
        enabled: false, mode: "refine", second_sampling_enabled: true, result_previews_enabled: false, denoise: 0.25, steps: 0,
        seed_mode: "inherit", seed_offset: 1, skip_fl2v: false,
        upscale_method: "lanczos", upscale_model: "",
        latent_upscale_model: "", latent_upscale_precision: "fp16", latent_upscale_device: "cuda",
        vsr_quality: "high",
        resolution_mode: "follow_director", aspect: "16:9", megapixels: 1,
        width: 1376, height: 768,
        rtx_deblur_enabled: false, rtx_deblur_quality: "medium", rtx_deblur_strength: 1,
    },
    preview: { enabled: true, preview_frames: 8, preview_fps: 12, max_resolution: 1024, jpeg_quality: 80, preview_every: 1 },
    save: {
        auto_save: false, filename_prefix: "video/MiniMaxH3_Director",
        format: "auto", codec: "auto", encoding: "auto", crf: 23,
    },
});

const clone = (value) => JSON.parse(JSON.stringify(value));
const inChoice = (value, choices, fallback) => choices.includes(String(value || "").toLowerCase()) ? String(value || "").toLowerCase() : fallback;

const POST_TEXT = {
    en: {
        global_title: "Global Refine", sampling: "Second Sampling",
        upscale: "Upscale", output_resolution: "Output Resolution",
        advanced: "Advanced Settings", enabled: "ON / OFF", disabled: "Disabled",
        runtime_detected: "Runtime detected (validated when generation starts)",
        missing_no_downgrade: "Not installed (stage will fail without downgrade)",
        lbh_note: "Director scans compatible MiniMax H3 learned-latent checkpoints from models/latent_upscale_models automatically. Select a checkpoint below; its 2D/3D architecture is detected from the weights. No separate LBH custom node is required.",
    },
    zh: {
        global_title: "全局精修", sampling: "二次采样",
        upscale: "放大", output_resolution: "输出分辨率",
        advanced: "高级设置", enabled: "开 / 关", disabled: "已停用",
        runtime_detected: "已检测到运行库（生成时验证）",
        missing_no_downgrade: "未安装（此阶段会失败，不会自动降级）",
        lbh_note: "Director 会自动扫描 models/latent_upscale_models 中兼容的 MiniMax H3 Learned Latent checkpoint。只需选择模型，2D/3D 架构会直接从权重自动识别；无需另装 LBH 自定义节点。",
    },
};

const POST_LABELS = {
    "global_refine.denoise": ["Denoise", "降噪强度"],
    "global_refine.steps": ["Steps (0=Auto)", "步数（0=自动）"],
    "global_refine.seed_mode": ["Refine Randomness", "精修随机性"],
    "global_refine.seed_offset": ["Seed Offset", "Seed 偏移"],
    "global_refine.result_previews_enabled": ["Pass Result Previews (extra VAE decode)", "阶段结果预览（额外 VAE 解码）"],
    "global_refine.skip_fl2v": ["Skip FL2V", "跳过 FL2V"],
    "global_refine.upscale_method": ["Method", "放大方法"],
    "global_refine.upscale_model": ["Model", "模型"],
    "global_refine.latent_upscale_model": ["H3 Latent Model", "H3 Latent 模型"],
    "global_refine.latent_upscale_precision": ["Precision", "精度"],
    "global_refine.latent_upscale_device": ["Device", "设备"],
    "global_refine.vsr_quality": ["VSR Quality", "VSR 质量"],
    "global_refine.resolution_mode": ["Resolution", "分辨率模式"],
    "global_refine.aspect": ["Aspect", "画幅比"],
    "global_refine.megapixels": ["Megapixels", "百万像素"],
    "global_refine.width": ["Width", "宽度"],
    "global_refine.height": ["Height", "高度"],
};

const POST_OPTION_LABELS = {
    "global_refine.seed_mode": {
        inherit: ["Keep original Seed (recommended)", "保持原 Seed（推荐）"],
        offset: ["Use Seed offset", "使用 Seed 偏移"],
    },
    "global_refine.upscale_method": {
        lanczos: ["Lanczos", "Lanczos"], upscale_model: ["Upscale Model", "放大模型"],
        nvidia_rtx_vsr: ["NVIDIA RTX VSR", "NVIDIA RTX VSR"],
        h3_learned_latent: ["H3 Learned Latent", "H3 Learned Latent"],
    },
    "global_refine.latent_upscale_precision": { fp16: ["FP16", "FP16"], bf16: ["BF16", "BF16"], fp32: ["FP32", "FP32"] },
    "global_refine.latent_upscale_device": { cuda: ["CUDA", "CUDA"], cpu: ["CPU", "CPU"] },
    "global_refine.vsr_quality": {
        low: ["Low", "Low"], medium: ["Medium", "Medium"], high: ["High", "High"], ultra: ["Ultra", "Ultra"],
    },
    "global_refine.resolution_mode": {
        follow_director: ["Follow Director", "跟随 Director"],
        aspect_megapixels: ["Aspect + Megapixels", "画幅比 + 百万像素"], custom: ["Custom", "自定义"],
    },
};

export function normalizePostprocessConfig(raw) {
    if (typeof raw === "string") {
        try { raw = raw.trim() ? JSON.parse(raw) : {}; } catch { raw = {}; }
    }
    raw = raw && typeof raw === "object" ? raw : {};
    const result = clone(DEFAULT_CONFIG);
    for (const key of ["global_refine", "preview", "save"]) {
        const legacy = key === "global_refine" ? raw.globalRefine : null;
        Object.assign(result[key], legacy || {}, raw[key] || {});
    }
    if (raw.liveTaePreview === false || raw.live_tae_preview === false) result.preview.enabled = false;
    const global = result.global_refine;
    global.enabled = !!global.enabled;
    global.mode = inChoice(global.mode, ["refine", "upscale"], "refine");
    global.second_sampling_enabled = global.second_sampling_enabled !== false;
    global.result_previews_enabled = global.result_previews_enabled === true;
    global.seed_mode = inChoice(global.seed_mode, ["inherit", "offset"], "inherit");
    const seedOffset = Number(global.seed_offset);
    global.seed_offset = Number.isFinite(seedOffset)
        ? Math.max(-2147483648, Math.min(2147483647, Math.trunc(seedOffset)))
        : 1;
    global.upscale_method = inChoice(global.upscale_method, ["lanczos", "upscale_model", "nvidia_rtx_vsr", "h3_learned_latent"], "lanczos");
    global.latent_upscale_model = String(global.latent_upscale_model || "").trim();
    delete global.latent_upscale_variant;
    global.latent_upscale_precision = inChoice(global.latent_upscale_precision, ["fp16", "bf16", "fp32"], "fp16");
    global.latent_upscale_device = inChoice(global.latent_upscale_device, ["cuda", "cpu"], "cuda");
    global.vsr_quality = inChoice(global.vsr_quality, ["low", "medium", "high", "ultra"], "high");
    global.resolution_mode = inChoice(global.resolution_mode, ["follow_director", "aspect_megapixels", "custom"], "follow_director");
    global.rtx_deblur_enabled = !!global.rtx_deblur_enabled;
    global.rtx_deblur_quality = inChoice(global.rtx_deblur_quality, ["low", "medium", "high", "ultra"], "medium");
    const deblurStrength = Number(global.rtx_deblur_strength);
    global.rtx_deblur_strength = Number.isFinite(deblurStrength) ? Math.max(0, Math.min(3, deblurStrength)) : 1;
    result.preview.enabled = result.preview.enabled !== false;
    result.save.auto_save = !!result.save.auto_save;
    result.save.filename_prefix = String(result.save.filename_prefix || "video/MiniMaxH3_Director").trim().slice(0, 512) || "video/MiniMaxH3_Director";
    result.save.format = String(result.save.format || "auto").trim().toLowerCase().slice(0, 32) || "auto";
    result.save.codec = String(result.save.codec || "auto").trim().toLowerCase().slice(0, 64) || "auto";
    result.save.encoding = result.save.encoding === "re-encode" ? "re-encode" : "auto";
    result.save.crf = Math.max(0, Math.min(51, Math.round(Number(result.save.crf) || 23)));
    result.version = 9;
    return result;
}

export function serializePostprocessConfig(raw) {
    return JSON.stringify(normalizePostprocessConfig(raw));
}

function snap(value) { return Math.max(32, Math.round(Number(value || 32) / 32) * 32); }

export function resolveGlobalTarget(config, directorWidth = 864, directorHeight = 480) {
    const global = normalizePostprocessConfig(config).global_refine;
    if (global.resolution_mode === "follow_director") return [snap(directorWidth), snap(directorHeight)];
    if (global.resolution_mode === "custom") return [snap(global.width), snap(global.height)];
    const match = String(global.aspect || "16:9").match(/^(\d+):(\d+)$/);
    const aw = Number(match?.[1] || 16), ah = Number(match?.[2] || 9);
    const scale = Math.sqrt(Math.max(0.1, Number(global.megapixels || 1)) * 1024 * 1024 / (aw * ah));
    return [snap(aw * scale), snap(ah * scale)];
}

export function setGlobalUpscaleEnabled(config, enabled) {
    const next = normalizePostprocessConfig(config);
    next.global_refine.mode = enabled ? "upscale" : "refine";
    return next;
}

export function globalRefineVisibility(config) {
    const global = normalizePostprocessConfig(config).global_refine;
    const upscaleEnabled = global.mode === "upscale";
    return {
        secondSampling: global.second_sampling_enabled,
        upscaleEnabled,
        seedOffset: global.second_sampling_enabled && global.seed_mode === "offset",
        upscaleModel: upscaleEnabled && global.upscale_method === "upscale_model",
        learnedLatent: upscaleEnabled && global.upscale_method === "h3_learned_latent",
        vsr: upscaleEnabled && global.upscale_method === "nvidia_rtx_vsr",
        aspectMegapixels: upscaleEnabled && global.resolution_mode === "aspect_megapixels",
        customSize: upscaleEnabled && global.resolution_mode === "custom",
    };
}

export function globalRefineSummary(config, width = 864, height = 480, locale = "en") {
    const global = normalizePostprocessConfig(config).global_refine;
    const zh = locale === "zh";
    if (!global.enabled) return POST_TEXT[zh ? "zh" : "en"].disabled;
    const parts = [];
    if (global.second_sampling_enabled) {
        const steps = Number(global.steps) > 0 ? `${global.steps} ${zh ? "步" : "Steps"}` : (zh ? "自动步数" : "Auto Steps");
        const seed = global.seed_mode === "offset"
            ? `${zh ? "Seed 偏移" : "Seed Offset"} ${Number(global.seed_offset) >= 0 ? "+" : ""}${Number(global.seed_offset)}`
            : (zh ? "保持原 Seed" : "Keep original Seed");
        parts.push(zh ? "二次采样 ON" : "Second Sampling ON", `D${Number(global.denoise).toFixed(2)}`, steps, seed);
        if (global.result_previews_enabled) parts.push(zh ? "阶段预览 ON" : "Pass Previews ON");
    } else {
        parts.push(zh ? "二次采样 OFF" : "Second Sampling OFF");
    }
    if (global.mode === "upscale") {
        const [targetW, targetH] = resolveGlobalTarget(config, width, height);
        let method = global.upscale_method === "upscale_model" ? (global.upscale_model || (zh ? "放大模型" : "Upscale Model")) : "Lanczos";
        if (global.upscale_method === "nvidia_rtx_vsr") {
            const quality = String(global.vsr_quality || "high");
            method = `RTX VSR ${quality.charAt(0).toUpperCase()}${quality.slice(1)}`;
        } else if (global.upscale_method === "h3_learned_latent") {
            method = "H3 Learned Latent";
        }
        parts.push(`${method} → ${targetW}×${targetH}`);
    }
    return parts.join(" · ");
}

export class PostprocessConfigStore {
    constructor(widget, { onChange } = {}) {
        this.widget = widget;
        this.onChange = onChange;
        this.listeners = new Set();
        this.value = normalizePostprocessConfig(widget?.value);
    }
    get() { return clone(this.value); }
    set(next, { notify = true } = {}) {
        this.value = normalizePostprocessConfig(next);
        const serialized = serializePostprocessConfig(this.value);
        if (this.widget) {
            this.widget.value = serialized;
            try {
                this.widget.callback?.(serialized);
            } catch (error) {
                console.warn("[MiniMax H3 Motion Director] postprocess widget callback failed:", error);
            }
        }
        if (notify) {
            try { this.onChange?.(this.get()); }
            catch (error) { console.warn("[MiniMax H3 Motion Director] postprocess onChange failed:", error); }
            this.listeners.forEach((listener) => {
                try { listener(this.get()); }
                catch (error) { console.warn("[MiniMax H3 Motion Director] postprocess listener failed:", error); }
            });
        }
        return this.get();
    }
    patch(section, key, value) {
        const next = this.get();
        next[section][key] = value;
        return this.set(next);
    }
    toggle(section) { return this.patch(section, "enabled", !this.value[section].enabled); }
    subscribe(listener) { this.listeners.add(listener); return () => this.listeners.delete(listener); }
    reload() {
        this.value = normalizePostprocessConfig(this.widget?.value);
        this.listeners.forEach((fn) => {
            try { fn(this.get()); }
            catch (error) { console.warn("[MiniMax H3 Motion Director] postprocess reload listener failed:", error); }
        });
    }
}

const STYLE_ID = "mmx-postprocess-styles";
function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.mmx-postprocess{display:grid;grid-template-columns:minmax(0,1fr);gap:10px;height:100%;min-height:0;box-sizing:border-box}
.mmx-post-column{min-width:0;overflow:auto;border:1px solid #343434;border-radius:8px;background:#181818;padding:10px;box-sizing:border-box}
.mmx-post-head,.mmx-post-section-head{display:flex;align-items:center;justify-content:space-between;gap:8px}.mmx-post-head{margin-bottom:8px}.mmx-post-head h3{margin:0;font-size:15px}
.mmx-post-section-head h4{margin:0;font-size:12px;color:#ddd}.mmx-post-enable,.mmx-post-subenable{display:flex;align-items:center;gap:6px;color:#4fff8f;font-weight:650}
.mmx-post-summary{min-height:18px;margin:0 0 8px;color:#aaa;font-size:11px}.mmx-post-note{margin:0 0 7px;color:#888;font-size:10px;line-height:1.45}
.mmx-post-section{margin:7px 0;padding:7px;border:1px solid #2d2d2d;border-radius:6px;background:#1d1d1d}.mmx-post-section>h4{margin:0 0 6px;font-size:12px;color:#ddd}
.mmx-post-section-body{margin-top:7px}.mmx-post-divider-title{margin:10px 0 6px;padding-top:8px;border-top:1px solid #303030;font-size:11px;font-weight:650;color:#bbb}
.mmx-post-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:6px 8px}.mmx-post-field{display:grid;grid-template-columns:minmax(82px,.9fr) minmax(0,1.2fr);align-items:center;gap:6px;min-width:0;font-size:11px;color:#aaa}
.mmx-post-field input,.mmx-post-field select{min-width:0;width:100%;height:26px;border:1px solid #3b3b3b;border-radius:4px;background:#242424;color:#ddd;padding:2px 5px;box-sizing:border-box}
.mmx-post-field input[type=checkbox]{width:auto;height:auto;justify-self:start}.mmx-post-wide{grid-column:1/-1}.mmx-post-conditional{min-width:0}.mmx-post-conditional[hidden],.mmx-post-section-body[hidden]{display:none!important}
.mmx-post-result{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:5px 7px;border-radius:4px;background:#202020}.mmx-post-result b{color:#ddd;font-size:12px}
.mmx-post-advanced>summary{cursor:pointer;color:#bbb;font-size:12px;font-weight:650;padding:3px}
.mmx-post-disabled{opacity:.52}.mmx-post-capability{font-size:10px;color:#999}.mmx-post-capability.bad{color:#ff8d8d}
@media(max-width:980px){.mmx-post-grid{grid-template-columns:1fr}}
`;
    document.head.appendChild(style);
}

function options(rows) { return rows.map(([value, label]) => `<option value="${value}">${label}</option>`).join(""); }
function field(label, path, type = "number", extra = "") {
    if (type === "select") return `<label class="mmx-post-field"><span data-field-label="${path}">${label}</span><select data-path="${path}">${extra}</select></label>`;
    if (type === "checkbox") return `<label class="mmx-post-field"><span data-field-label="${path}">${label}</span><input type="checkbox" data-path="${path}"></label>`;
    return `<label class="mmx-post-field"><span data-field-label="${path}">${label}</span><input type="${type}" data-path="${path}" ${extra}></label>`;
}
function conditional(name, content) { return `<div class="mmx-post-conditional" data-conditional="${name}">${content}</div>`; }

export function mountPostprocessUI(container, store, { fetchApi, directorSize = () => [864, 480], locale = () => "zh" } = {}) {
    ensureStyles();
    const root = document.createElement("div");
    root.className = "mmx-postprocess";
    root.innerHTML = `
      <section class="mmx-post-column" data-section="global_refine">
        <div class="mmx-post-head"><h3 data-post-text="global_title">全局精修</h3><label class="mmx-post-enable"><input type="checkbox" data-path="global_refine.enabled"> <span data-post-text="enabled">ON / OFF</span></label></div>
        <p class="mmx-post-summary" data-summary="global_refine"></p>
        <div class="mmx-post-section" data-second-sampling-section>
          <div class="mmx-post-section-head"><h4 data-post-text="sampling">二次采样</h4><label class="mmx-post-subenable"><input type="checkbox" data-path="global_refine.second_sampling_enabled"> <span data-post-text="enabled">ON / OFF</span></label></div>
          <div class="mmx-post-section-body" data-second-sampling-body><div class="mmx-post-grid">
            ${field("Denoise", "global_refine.denoise", "number", 'min="0.01" max="1" step="0.01"')}
            ${field("Steps (0=Auto)", "global_refine.steps", "number", 'min="0" max="200" step="1"')}
            ${field("Refine Randomness", "global_refine.seed_mode", "select", options([["inherit","Keep original Seed (recommended)"],["offset","Use Seed offset"]]))}
            ${conditional("seed_offset", field("Seed Offset", "global_refine.seed_offset", "number", 'min="-2147483648" max="2147483647" step="1"'))}
            ${field("Pass Result Previews (extra VAE decode)", "global_refine.result_previews_enabled", "checkbox")}
            ${field("Skip FL2V", "global_refine.skip_fl2v", "checkbox")}
          </div></div>
        </div>
        <div class="mmx-post-section" data-upscale-section>
          <div class="mmx-post-section-head"><h4 data-post-text="upscale">Upscale</h4><label class="mmx-post-subenable"><input type="checkbox" data-upscale-enabled> <span data-post-text="enabled">ON / OFF</span></label></div>
          <div class="mmx-post-section-body" data-upscale-body>
            <div class="mmx-post-grid">
              ${field("Method", "global_refine.upscale_method", "select", options([["lanczos","Lanczos"],["upscale_model","Upscale Model"],["nvidia_rtx_vsr","NVIDIA RTX VSR"],["h3_learned_latent","H3 Learned Latent"]]))}
              ${conditional("upscale_model", field("Model", "global_refine.upscale_model", "select", '<option value="">—</option>'))}
              ${conditional("learned_latent", field("H3 Latent Model", "global_refine.latent_upscale_model", "select", '<option value="">—</option>'))}
              ${conditional("learned_latent", field("Precision", "global_refine.latent_upscale_precision", "select", options([["fp16","FP16"],["bf16","BF16"],["fp32","FP32"]])))}
              ${conditional("learned_latent", field("Device", "global_refine.latent_upscale_device", "select", options([["cuda","CUDA"],["cpu","CPU"]])))}
              <p class="mmx-post-note mmx-post-wide" data-conditional="learned_latent" data-post-text="lbh_note"></p>
              ${conditional("vsr_quality", field("VSR Quality", "global_refine.vsr_quality", "select", options([["low","Low"],["medium","Medium"],["high","High"],["ultra","Ultra"]])))}
              <div class="mmx-post-capability mmx-post-wide" data-conditional="vsr_status" data-capability="nvidia_rtx_vsr"></div>
            </div>
            <div class="mmx-post-divider-title" data-post-text="output_resolution">Output Resolution</div>
            <div class="mmx-post-grid">
              ${field("Resolution", "global_refine.resolution_mode", "select", options([["follow_director","Follow Director"],["aspect_megapixels","Aspect + Megapixels"],["custom","Custom"]]))}
              ${conditional("aspect", field("Aspect", "global_refine.aspect", "select", options([["1:1","1:1"],["4:3","4:3"],["3:4","3:4"],["16:9","16:9"],["9:16","9:16"],["21:9","21:9"]])))}
              ${conditional("megapixels", field("Megapixels", "global_refine.megapixels", "number", 'min="0.1" max="16" step="0.1"'))}
              ${conditional("width", field("Width", "global_refine.width", "number", 'min="32" max="8192" step="32"'))}
              ${conditional("height", field("Height", "global_refine.height", "number", 'min="32" max="8192" step="32"'))}
              <div class="mmx-post-result mmx-post-wide"><span data-field-label="resolved_target">Final Size</span><b data-resolved-target></b></div>
            </div>
          </div>
        </div>
      </section>`;
    container.replaceChildren(root);

    const readInput = (element) => element.type === "checkbox" ? element.checked
        : element.type === "number" ? Number(element.value) : element.value;
    root.addEventListener("change", (event) => {
        const target = event.target;
        if (target?.matches?.("[data-upscale-enabled]")) {
            store.set(setGlobalUpscaleEnabled(store.get(), target.checked));
            return;
        }
        const input = target?.closest?.("[data-path]") || target;
        const path = input?.dataset?.path;
        if (!path) return;
        const [section, key] = path.split(".");
        store.patch(section, key, readInput(input));
    });
    let capabilities = null;
    const setConditional = (name, hidden) => {
        root.querySelectorAll(`[data-conditional="${name}"]`).forEach((element) => { element.hidden = hidden; });
    };
    const updateLocale = (language = locale()) => {
        const lang = language === "en" ? "en" : "zh";
        root.querySelectorAll("[data-post-text]").forEach((element) => {
            element.textContent = POST_TEXT[lang][element.dataset.postText] || element.textContent;
        });
        root.querySelectorAll("[data-field-label]").forEach((element) => {
            const pair = element.dataset.fieldLabel === "resolved_target" ? ["Final Size", "最终尺寸"] : POST_LABELS[element.dataset.fieldLabel];
            if (pair) element.textContent = pair[lang === "zh" ? 1 : 0];
        });
        root.querySelectorAll("select[data-path]").forEach((select) => {
            const labels = POST_OPTION_LABELS[select.dataset.path];
            if (!labels) return;
            for (const option of select.options) {
                const pair = labels[option.value];
                if (pair) option.textContent = pair[lang === "zh" ? 1 : 0];
            }
        });
        const vsr = root.querySelector('[data-capability="nvidia_rtx_vsr"]');
        if (vsr && capabilities) {
            const ready = !!capabilities.dependencies?.nvidia_rtx_vsr;
            vsr.textContent = `RTX VSR: ${POST_TEXT[lang][ready ? "runtime_detected" : "missing_no_downgrade"]}`;
        }
        render(store.get());
    };
    const render = (config) => {
        root.querySelectorAll("[data-path]").forEach((input) => {
            const [section, key] = input.dataset.path.split(".");
            const value = config[section]?.[key];
            if (input.type === "checkbox") input.checked = !!value;
            else if (String(input.value) !== String(value ?? "")) input.value = value ?? "";
        });
        const visible = globalRefineVisibility(config);
        root.querySelector("[data-second-sampling-body]").hidden = !visible.secondSampling;
        root.querySelector("[data-upscale-enabled]").checked = visible.upscaleEnabled;
        root.querySelector("[data-upscale-body]").hidden = !visible.upscaleEnabled;
        setConditional("seed_offset", !visible.seedOffset);
        setConditional("upscale_model", !visible.upscaleModel);
        setConditional("learned_latent", !visible.learnedLatent);
        setConditional("vsr_quality", !visible.vsr);
        setConditional("vsr_status", !visible.vsr);
        setConditional("aspect", !visible.aspectMegapixels);
        setConditional("megapixels", !visible.aspectMegapixels);
        setConditional("width", !visible.customSize);
        setConditional("height", !visible.customSize);
        const [w, h] = directorSize();
        root.querySelector('[data-summary="global_refine"]').textContent = globalRefineSummary(config, w, h, locale());
        const [tw, th] = resolveGlobalTarget(config, w, h);
        root.querySelector("[data-resolved-target]").textContent = `${tw}×${th}`;
        root.querySelector('[data-section="global_refine"]').classList.toggle("mmx-post-disabled", !config.global_refine.enabled);
    };
    const unsubscribe = store.subscribe(render);
    render(store.get());
    updateLocale(locale());

    if (fetchApi) fetchApi("/minimax/bb/motion-director/postprocess_capabilities").then((response) => response.json()).then((caps) => {
        const fill = (path, rows) => {
            const select = root.querySelector(`[data-path="${path}"]`);
            const current = select.value;
            select.innerHTML = '<option value="">—</option>' + (rows || []).map((name) => `<option value="${name}">${name}</option>`).join("");
            select.value = current;
        };
        fill("global_refine.upscale_model", caps.upscale_models);
        fill("global_refine.latent_upscale_model", caps.latent_upscale_models);
        const h3LatentModels = caps.latent_upscale_models || [];
        const currentGlobal = store.get().global_refine;
        if (!currentGlobal.latent_upscale_model && h3LatentModels.length === 1) {
            store.patch("global_refine", "latent_upscale_model", h3LatentModels[0]);
        }
        capabilities = caps;
        const vsr = root.querySelector('[data-capability="nvidia_rtx_vsr"]');
        const ready = !!caps.dependencies?.nvidia_rtx_vsr;
        vsr.classList.toggle("bad", !ready);
        updateLocale(locale());
    }).catch(() => {});
    return { root, render, updateLocale, destroy: unsubscribe };
}

export { DEFAULT_CONFIG };
