"""Shared filesystem-safe cache path helpers for ComfyUI node IDs."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from pathlib import Path


def cache_node_dir_name(node_id: object) -> str:
    """Convert ComfyUI compound node IDs into Windows-safe directory names."""
    return str(node_id).replace(":", "_")


def cache_root(output_directory: str | Path, cache_name: str, node_id: object) -> Path:
    """Create and return the cache directory for one Director node."""
    root = Path(output_directory) / cache_name / cache_node_dir_name(node_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def safe_unlink(path: Path) -> bool:
    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
        return True
    except OSError:
        return False


def atomic_publish(tmp: Path, dest: Path) -> None:
    """Move ``tmp`` to ``dest``, tolerating clouds that block same-name overwrite."""
    try:
        os.replace(tmp, dest)
        return
    except OSError:
        pass
    # Some cloud mounts reject overwrite of an existing name; remove then rename.
    safe_unlink(dest)
    try:
        os.replace(tmp, dest)
        return
    except OSError:
        pass
    try:
        tmp.rename(dest)
        return
    except OSError:
        # Last resort: keep the unique temp as the published file name is blocked.
        # Caller may still fail if even create-new is denied.
        raise


def _write_via_temp(dest: Path, write_fn: Callable[[Path], None]) -> None:
    """Write to a unique temp name in the same folder, then publish to ``dest``.

    Kept under its original private name because the motion-context and latent-context
    caches import it; the full decoded segment cache it belonged to is gone.
    """
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")
    try:
        write_fn(tmp)
        atomic_publish(tmp, dest)
    finally:
        safe_unlink(tmp)
