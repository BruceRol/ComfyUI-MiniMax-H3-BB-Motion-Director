from pathlib import Path


def test_postprocess_module_boot_token_is_v10():
    timeline = Path("web/js/minimax_timeline.js").read_text(encoding="utf-8")
    assert 'minimax_postprocess_ui.mjs?boot=postprocess_output_v10' in timeline
