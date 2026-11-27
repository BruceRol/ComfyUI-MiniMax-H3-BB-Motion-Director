# Step 3 plan — per-clip MP4 streaming, retention windows, Low RAM

Status: **implemented, awaiting manual verification in ComfyUI.**
Implemented: `director/clip_writer.py` (writer), `director/clip_window.py` (`ClipWindow`),
`director/low_ram.py` (policy), `director/clip_stream.py` (`ClipStreamState`),
`director/clip_merge.py` (merge), the window-aware continuity and Source Bridge ops, the
rewritten interleaved legacy loop, the Low RAM node input and its frontend wiring, the node
contract (`OUTPUT_NODE = True`, no public outputs), the frontend Output group, and the
Face Refine removal. Remaining: the manual ComfyUI run, then the user guide rewrite.

## Locked requirements

1. Output is h264-mp4 only, one file per clip, named `clip-XXX.mp4` numbered from 1 like the
   timeline prompts (the first clip is `clip-001.mp4`).
2. No metadata, no pingpong, no `trim_to_audio`.
3. Subdirectory from `dirname_prefix` inside the output folder.
4. Every clip has audio (silent padded track when the segment has none). No MP4 without audio.
5. No PNG export at all.
6. Export mode `all`: every segment is written, then merged into `full_video.mp4` and the
   clip files are deleted. Export mode `by segment`: only the selected segments are written,
   no merge.
7. Console prints the full path of every MP4 written, and of the merged file.
8. Clips are written **incrementally during generation**, as soon as each clip is final
   (its boundary with the next segment has been applied).
9. "Low RAM" option in the Performance group. ON = write the middle immediately, hold the
   head/tail windows in RAM. OFF = keep everything in RAM. **No cache files are written in
   either mode** (see "Cross-run regeneration is dropped").
10. Reference implementations: VideoHelperSuite `VHS_VideoCombine`, comfy-core `SaveVideo`.

## Key facts established during the design discussion

- **Storage location does not change correctness.** RAM or disk, the ops receive the same
  tensor. The only requirement is that the frames a later op needs are *reachable*.
  So windows are purely a memory optimization, and they must be sized to cover the
  **decision region**, not just the write region.
- **Source Bridge does not prepend frames.** `assemble_source_bridges` overwrites in place:
  left at `frame_count - 1`, right at offsets 0 and 1. Clip lengths are unchanged, so no
  prefix region is needed in the window.
- **Order at a boundary** (today): `assemble_source_bridges` runs before
  `concat_continuous_chunks`, so bridge writes happen first and the pair op may overwrite the
  bridge frame at left `n-1`. The rewrite must preserve this order.
- **Pair-op order per boundary** in `concat_continuous_chunks`:
  `_unfreeze_held_tail(left)` → `_break_hold_pop_window(left, from_end=True)` (flag-gated) →
  `_break_hold_pop_window(chunks[i], from_end=False)` → `_ease_opening_spikes` →
  `_soften_body0_toward_prev` → `_additive_opening_luma` → `_micro_seam_bridge`.
- **Left-side ops anchor at the end** of the previous chunk, **right-side ops anchor at the
  start** of the next chunk. `_break_hold_pop_window(from_end=True)` scans the last
  `min(scan, n - 2)` frames (`start = max(0, n - 1 - min(scan, n - 2))`), so the tail window
  must cover the scan, not only the write.
- **Continuity writes land inside the windows**, so the ops can run directly on the window
  views in place; `set_frame_abs` is not needed for writes. Views must be re-assigned after
  every op because ops may return clones (`set_left_view` / `set_right_view`).
- **`_run_one_segment` returns a full chunk**, so the full chunk exists at capture time. The
  saving is across iterations, not within one.
- **When `context_pipeline_active` or `bridge_feature_active` is true**, the assembly is a plain
  `cat_frames_variable_size` with **no pair ops**: each chunk is final at capture.
- `CONTINUITY_HOLD_MAX_FRAMES = 0` and `CONTINUITY_HOLD_POP_ON_TAIL = False`, so today only
  `_micro_seam_bridge` can write the previous chunk (its last 3 frames). The rewrite must stay
  correct if those flags change.

## Window shape (Low RAM ON)

`ClipWindow` = `head` (first `min(12, n)`) + `middle` + `tail` (last `min(8, n - head_len)`).
`retained` = `cat(head, tail)`; when `middle` is empty (short chunk or Low RAM OFF) the whole
chunk is retained and `head`/`tail` are views into it.

Reach check (decision region, not write region):

| Op | Side | Decision region | Write region | Window |
|---|---|---|---|---|
| `_unfreeze_held_tail` | left | last `min(hold_max, n)` | last `min(hold_max, n)` | tail |
| `_break_hold_pop_window(from_end=True)` | left | last `min(6, n-2)` pairs, `start = n-1-min(6,n-2)` | same | tail (8 ⇒ `start` = window index 1 = full-chunk `n-7`) |
| `_break_hold_pop_window(from_end=False)` | right | first `min(6, n-2)` | first `min(6, n-2)` | head |
| `_ease_opening_spikes` | right | first `min(5, n)` | first `min(5, n)` | head |
| `_soften_body0_toward_prev` | right | frame 0 | frame 0 | head |
| `_additive_opening_luma` | right | `guide[-1]` + body `0..min(12, n)-1` | body `0..min(12, n)-1` | head (12) — exact, no guard frame needed |
| `_micro_seam_bridge` | left | last `min(3, n)` | last `min(3, n)` | tail |
| `_micro_seam_bridge` | right | first `min(3, n)` | first `min(3, n)` | head |
| Source Bridge | left | anchor `n-3`, writes `n-1` | — | tail |
| Source Bridge | right | anchor offset 2, writes 0,1 | — | head |

- `HEAD_WINDOW_FRAMES = 12` — set by `_additive_opening_luma` (the widest left-edge region).
- `TAIL_WINDOW_FRAMES = 8` — `scan(6) + 1` guard frame; covers `_micro_seam_bridge` (3).
- Short chunks (`n < head + tail`) keep the whole chunk.

**Live peak** = one full chunk (current segment) + previous tail window (8 frames).
At 1280x720 float32 NHWC: chunk ≈ 63.3 MiB for 60 frames, tail window ≈ 8.4 MiB.
Full retention would be 633 MiB for 10 segments, 1.24 GiB for 20, 2.47 GiB for 40.
Windows reduce the retention part to ≈ 21 MiB/segment, live peak ≈ 8.4 MiB.

## Target loop (single pass, Low RAM ON)

```
for seg in run order:
    chunk = _run_one_segment(seg)                 # full chunk exists here
    window = ClipWindow(seg, chunk, low_ram=True)
    if previous window exists:
        apply Source Bridge for (prev, seg)       # bridge first, as today
        apply the continuity pair op              # may overwrite the bridge frame at prev's n-1
        finalize the previous clip                # head + middle + tail, in order
        write the previous clip                   # console prints the path
    apply the right-side pair ops on this window  # only when no bridge/context pipeline
    write window.head + window.middle             # immediately
    hold window.tail                              # needed by the next boundary
```

At the end: finalize + write the last clip, then merge in export mode `all`.

Low RAM OFF: `write_ready()` is a no-op, `finalize()` writes the whole chunk, nothing is
written until the end, and the merge happens at the end in export mode `all`.

## Cross-run regeneration is dropped

Cross-run partial regeneration (reusing a previous run's segment cache) is removed entirely;
it may be rebuilt later as a separate system. Consequences:

- In export mode `by segment`, a selected segment that references an earlier segment forces
  that producer to be regenerated in the same run.
- No segment cache files are written or deleted in either Low RAM mode; Low RAM is purely a
  windowing policy. `director/mixed_selection.py`, `director/cache_policy.py` and
  `director/segment_cache.py` are deleted, `load_segment_cache` is gone from
  `director/executor_core.py`, and `resolve_prev_segment_output` raises an explicit error when
  a referenced segment was not produced in this run.
- Previews are sent during generation from the full chunk while it is in memory, so the
  Source Bridge preview no longer needs a disk reload.
- The Motion Context disk cache is removed as well.  `director/context_cache.py`,
  `director/latent_context_cache.py`, `director/cache_path.py` and `director/context_identity.py`
  are deleted and `plan.cache_settings` is gone: Motion Context is an in-run handoff only
  (`director/context_store.py`).  A selected segment whose predecessor was not generated in this
  run raises an explicit error instead of reading a stale file.
- The retained context is sized from the `context_length` widget through
  `low_ram.context_tail_frames`, never from the fixed `MAX_PERSISTED_CONTEXT_FRAMES = 39`, so
  Low RAM ON and OFF hand off exactly the same window.  The report section `Cache` becomes
  `Segments` (`generated in this run` / `not part of this run`).

## Implementation checklist

- [x] `director/low_ram.py`: `resolve_low_ram(value, default=True)`, `window_sizes(chunk_frames)`.
- [x] `director/clip_window.py`: `ClipWindow` (head/middle/tail, absolute accessors, views,
      `write_ready()`, `finalize()`, `write_audio()`, Low RAM flag, whole-chunk fallback).
- [x] `director/segment_continuity.py`: `apply_pair_ops_windows(left_window, right_window, plan)`,
      mirroring `concat_continuous_chunks` on the views (re-assign each view after every op).
- [x] `director/source_bridge.py`: window-aware write (`apply_source_bridge_to_windows`) writing
      left `n-1` and right `0,1`; `bridge_anchors_windows` reads the anchors through the views;
      `assemble_source_bridges` validates `frames.shape[0] == seg.frame_count` and cannot run on a
      window tensor.
- [x] `director/clip_stream.py`: `ClipStreamState` with `open_window` / `write_ready` /
      `finalize` / `finalize_open` around `ClipWindow`; `export_order()` drives the merge.
- [x] Rewrite the legacy segment loop as the single pass above, with the Source Bridge loop
      interleaved, and wire `capture_mixed_reference_frames(plan, seg, chunk)` at the capture point.
- [x] Low RAM option: node input in `director_perf_inputs()`, frontend Performance group layout,
      i18n labels/tooltips, `DIRECTOR_GROUP_WIDGET_LAYOUT` Experimental index 21 -> 22.
- [x] `director/clip_merge.py`: concat demuxer with `-c copy` using `lib/audio_io._ffmpeg_bin()`,
      re-encode fallback on param mismatch, rename for a single clip, delete clips only on
      success, `faststart` on the merged file, console path logging.
- [x] Face Refine removal (confirmed scope): delete `face_refine_pipeline.py`,
      `face_refine_streaming.py`, `face_refine_validation.py`, `face_stitch.py`, `face_track.py`;
      delete the tests; delete the SAM-model HTTP route; remove the `face_refine` block from the
      `postprocess_config` schema, the `face_refine` phase in `director/progress.py`, the Face
      Refine column and `faceRefineVisibility`/`faceRefineSummary` in
      `web/js/minimax_postprocess_ui.mjs`, `run_face` / `_aggregate_face_outcome` /
      `assembled_face_noop` in `director/executor_core.py`. No no-op hook is needed: the rewritten
      loop never calls `apply_face_refine`, and the report section is now "Clip Output".
- [x] `director/context_store.py`: `RetainedMotionContext` and `prepare_latent_context_tail(latent,
      handoff, span=...)`; `executor_core_legacy.py` reads Motion Context only from the in-memory
      retained windows, the latent tail is sized with `low_ram.context_tail_frames(requested_context)`,
      `tests/test_visual_latent_validity.py` is retargeted to the store, the dead cache helpers
      (`color_reanchor_cache_settings`, `postprocess_cache_fingerprint`) are removed, and
      `minimax_motion_context_cache/` is dropped from `.gitignore`.
- [ ] Manual verification in ComfyUI (full restart, hard browser refresh, recreate the Director
      node, workflow run). Then commit/push Step 3, excluding `AGENTS.md` and the personal
      `start_projet_*` / `stop-opencode.*` scripts.
- [ ] Rewrite `docs/USER_GUIDE.md` (still describes Inputs/Assets nodes, optional external
      sampling, Final Result panel, final video export).
