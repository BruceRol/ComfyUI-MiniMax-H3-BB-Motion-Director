from director.postprocess_config import (
    POSTPROCESS_CONFIG_VERSION,
    normalize_postprocess_config,
    postprocess_config_identity,
)


def test_postprocess_config_version_is_v10_and_face_refine_is_gone():
    assert POSTPROCESS_CONFIG_VERSION == 10
    normalized = normalize_postprocess_config(
        {
            "version": 9,
            "face_refine": {"enabled": True, "mask_mode": "sam", "sam_model": "sam2_t.pt"},
            "global_refine": {"result_previews_enabled": True},
        }
    )
    assert normalized["version"] == 10
    assert "face_refine" not in normalized
    assert normalized["global_refine"]["result_previews_enabled"] is True


def test_result_preview_does_not_change_config_identity():
    base = {
        "version": 9,
        "global_refine": {"enabled": True, "result_previews_enabled": False},
    }
    preview = {
        **base,
        "global_refine": {"enabled": True, "result_previews_enabled": True},
    }
    changed_refine = {
        **base,
        "global_refine": {"enabled": True, "result_previews_enabled": False, "denoise": 0.55},
    }
    assert postprocess_config_identity(base) == postprocess_config_identity(preview)
    assert postprocess_config_identity(base) != postprocess_config_identity(changed_refine)
