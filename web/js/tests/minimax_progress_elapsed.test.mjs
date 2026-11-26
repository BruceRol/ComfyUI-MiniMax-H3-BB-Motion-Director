import assert from "node:assert/strict";
import fs from "node:fs";
const src = fs.readFileSync(new URL("../minimax_timeline.js", import.meta.url), "utf8");
assert.match(src, /elapsed_seconds/);
assert.match(src, /phase_elapsed_seconds/);
assert.match(src, /Elapsed/);
assert.match(src, /Phase/);
console.log("progress elapsed UI test passed");
