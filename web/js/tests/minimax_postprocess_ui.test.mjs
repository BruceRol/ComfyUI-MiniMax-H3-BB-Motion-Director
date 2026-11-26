import assert from "node:assert/strict";
import fs from "node:fs";
import {globalRefineSummary,globalRefineVisibility,normalizePostprocessConfig,setGlobalUpscaleEnabled} from "../minimax_postprocess_ui.mjs";

let cfg=normalizePostprocessConfig({global_refine:{enabled:true,mode:"refine",seed_mode:"inherit",resolution_mode:"follow_director",upscale_method:"lanczos"}});
assert.equal(cfg.version,9);
assert.deepEqual(globalRefineVisibility(cfg),{secondSampling:true,upscaleEnabled:false,seedOffset:false,upscaleModel:false,learnedLatent:false,vsr:false,aspectMegapixels:false,customSize:false});
assert.match(globalRefineSummary(cfg, 1376, 768), /Keep original Seed/);
cfg=setGlobalUpscaleEnabled(cfg,true); cfg.global_refine.upscale_method="nvidia_rtx_vsr"; cfg.global_refine.vsr_quality="high"; cfg.global_refine.resolution_mode="aspect_megapixels";
assert.equal("vsr_source" in cfg.global_refine,false);
assert.match(globalRefineSummary(cfg, 1376, 768), /RTX VSR High/);

const latent=normalizePostprocessConfig({global_refine:{enabled:true,mode:"upscale",upscale_method:"h3_learned_latent",latent_upscale_model:"h3.safetensors",latent_upscale_variant:"3D",latent_upscale_precision:"BF16",latent_upscale_device:"CPU"}});
assert.equal(latent.global_refine.upscale_method,"h3_learned_latent");
assert.equal(latent.global_refine.latent_upscale_model,"h3.safetensors");
assert.equal("latent_upscale_variant" in latent.global_refine,false);
assert.equal(latent.global_refine.latent_upscale_precision,"bf16");
assert.equal(latent.global_refine.latent_upscale_device,"cpu");
assert.equal(globalRefineVisibility(latent).learnedLatent,true);
assert.match(globalRefineSummary(latent, 1376, 768), /H3 Learned Latent →/);
assert.doesNotMatch(globalRefineSummary(latent, 1376, 768), /\b2D\b|\b3D\b/);

const uiSource=fs.readFileSync(new URL("../minimax_postprocess_ui.mjs",import.meta.url),"utf8");
assert.doesNotMatch(uiSource,/Requires the separately installed LBH/);
assert.match(uiSource,/No separate LBH custom node is required/);
assert.match(uiSource,/architecture is detected from the weights/);
assert.doesNotMatch(uiSource,/field\("Latent Variant",\s*"global_refine\.latent_upscale_variant"/);
assert.doesNotMatch(uiSource,/2D \+ Temporal \(recommended\)/);

// Face Refine is gone: the config schema no longer carries a face section and
// legacy saved workflows that mention it are ignored.
assert.equal("face_refine" in normalizePostprocessConfig({face_refine:{enabled:true}}),false);
assert.doesNotMatch(uiSource,/face_refine|faceRefine|Face Refine/);
console.log("postprocess UI state tests passed");
