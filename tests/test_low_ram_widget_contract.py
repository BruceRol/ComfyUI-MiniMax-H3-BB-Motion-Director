from pathlib import Path


def test_low_ram_is_a_performance_widget_on_both_sides():
    common = Path("nodes/director_common.py").read_text(encoding="utf-8")
    node = Path("nodes/director.py").read_text(encoding="utf-8")

    perf = common.split("def director_perf_inputs", 1)[1]
    assert '"low_ram"' in perf
    assert '"default": True' in perf.split('"low_ram"', 1)[1][:200]

    # The node accepts it and hands it to the executor core, which is the only place
    # the retention policy is resolved.
    assert "low_ram=True," in node
    assert "low_ram=low_ram," in node


def test_low_ram_is_wired_in_the_frontend_performance_group():
    timeline = Path("web/js/minimax_timeline.js").read_text(encoding="utf-8")
    sections = Path("web/js/minimax_director_sections.js").read_text(encoding="utf-8")
    runtime_fix = Path("web/js/zz_minimax_director_runtime_fix.js").read_text(encoding="utf-8")
    i18n = Path("web/js/minimax_i18n.js").read_text(encoding="utf-8")

    assert 'const PERF_WIDGET_ORDER = ["bd_grp_perf", "clear_vram_between_segments", "low_ram"]' in timeline
    assert 'low_ram: "widget.lowRam"' in timeline
    assert 'low_ram: "widget.tooltip.lowRam"' in timeline

    # Label + section frame, and the runtime-fix label source.
    assert '"low_ram",' in sections.split("FIXED_LABEL_WIDGETS", 1)[1].split("]);", 1)[0]
    perf_block = sections.split("const groups = [", 1)[1].split('"bd_grp_perf"', 1)[1].split("]", 1)[0]
    assert '"low_ram",' in perf_block
    assert 'low_ram: "low_ram"' in runtime_fix

    for key in ("widget.lowRam", "widget.tooltip.lowRam"):
        assert i18n.count(f'"{key}"') == 2, f"{key} must exist in both the zh and en tables"


def test_experimental_group_widget_index_shifts_with_the_new_perf_widget():
    # Indices include the control_after_generate slot, so adding low_ram after
    # clear_vram_between_segments moves the Experimental group header from 21 to 22.
    sampling = Path("web/js/minimax_sampling_ui.js").read_text(encoding="utf-8")
    layout = sampling.split("DIRECTOR_GROUP_WIDGET_LAYOUT", 1)[1].split("]);", 1)[0]
    for expected in ('[2, "采样设置"]', '[12, "Motion Context"]', '[18, "高级采样"]', '[19, "性能"]', '[22, "Experimental"]'):
        assert expected in layout
