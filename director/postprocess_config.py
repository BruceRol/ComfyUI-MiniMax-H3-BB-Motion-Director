"""v10 postprocess compatibility facade.

The v9 implementation is preserved verbatim in ``postprocess_config_legacy``.
This facade keeps its public API stable; Face Refine has been removed from the
config schema.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from . import postprocess_config_legacy as _legacy
from .postprocess_config_legacy import *  # noqa: F401,F403

POSTPROCESS_CONFIG_VERSION = 10
DEFAULT_POSTPROCESS_CONFIG = copy.deepcopy(_legacy.DEFAULT_POSTPROCESS_CONFIG)
DEFAULT_POSTPROCESS_CONFIG["version"] = POSTPROCESS_CONFIG_VERSION


def normalize_postprocess_config(raw: Any) -> dict[str, Any]:
    result = _legacy.normalize_postprocess_config(raw)
    result["version"] = POSTPROCESS_CONFIG_VERSION
    return result


def serialize_postprocess_config(raw: Any) -> str:
    return json.dumps(
        normalize_postprocess_config(raw),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def __getattr__(name: str):
    return getattr(_legacy, name)


__all__ = list(_legacy.__all__)
for _name in (
    "DEFAULT_POSTPROCESS_CONFIG",
    "POSTPROCESS_CONFIG_VERSION",
    "normalize_postprocess_config",
    "serialize_postprocess_config",
    "postprocess_config_identity",
):
    if _name not in __all__:
        __all__.append(_name)
