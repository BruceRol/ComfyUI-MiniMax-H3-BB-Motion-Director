# Portions derived from ComfyUI_MiniMaxH3_Director
# Copyright AIMixer and contributors
# Originally licensed under Apache License 2.0
# Modified for MiniMax H3 Motion Director, 2026-08-09
# This derivative project is distributed under GPL-3.0.
# See NOTICE and LICENSES/Apache-2.0-AIMixer.txt.

"""Standard external SAMPLER/SIGMAS sampling path (internal sampling removed)."""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch

log = logging.getLogger("ComfyUI-MiniMax-H3-Motion-Director.director.core_sampling")

PhaseCallback = Callable[[str, float], None]
StepPreviewCallback = Callable[[int, int, Any, Any], None]


class _DirectorPreviewOuterSample:
    """Observe packed sampler x0 together with ComfyUI's authoritative shapes."""

    def __init__(self, callback: StepPreviewCallback, every: int):
        self.callback = callback
        self.every = max(1, int(every))

    def __call__(
        self,
        executor,
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask,
        callback,
        disable_pbar,
        seed,
        latent_shapes,
    ):
        original_callback = callback

        def observed(step, x0, x, total_steps):
            try:
                if step % self.every == 0 or step >= max(0, int(total_steps) - 1):
                    # x0 is observation-only.  Never rebind, reshape in-place, or
                    # mutate it: ComfyUI reuses this packed tensor downstream.
                    self.callback(int(step), int(total_steps), x0, latent_shapes)
            except Exception as exc:
                log.warning("Director step preview skipped; sampling continues: %s", exc)
            if original_callback is not None:
                original_callback(step, x0, x, total_steps)

        return executor(
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            observed,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )


def _install_preview_outer_wrapper(model, callback, every: int):
    if callback is None:
        return model, False
    try:
        import comfy.patcher_extension

        observed_model = model.clone()
        observed_model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "minimax_motion_director_preview",
            _DirectorPreviewOuterSample(callback, every),
        )
        return observed_model, True
    except Exception as exc:
        # Older ComfyUI can still use the callback fallback, but current packed
        # H3 builds require OUTER_SAMPLE to expose latent_shapes.
        log.warning("Director OUTER_SAMPLE preview hook unavailable; using callback fallback: %s", exc)
        return model, False


def describe_external_sampler(sampler) -> str:
    fn = getattr(sampler, "sampler_function", None)
    if fn is None:
        fn = getattr(sampler, "sample", None)
    name = getattr(fn, "__name__", None)
    if name:
        return name
    return type(sampler).__name__


def resolve_refine_model(refine_model, fallback_model):
    """Resolve the model a refine pass runs on; a connected model wins.

    Without a connected ``refine_model`` the pass follows the first-pass model.
    A connected object is already a patched model, so it carries the sigma range
    its own SIGMAS were built from.
    """
    if refine_model is None:
        return fallback_model, "Follow First Pass"

    import comfy.model_base

    base_model = getattr(refine_model, "model", None)
    if not isinstance(base_model, comfy.model_base.MiniMaxH3):
        raise ValueError(
            "Refine model must be MiniMax H3; the connected refine_model is "
            "%s." % type(base_model).__name__
        )
    return refine_model, str(getattr(refine_model, "model_name", "") or "connected refine_model")


def resolve_sampling_mode(sampler, sigmas) -> str:
    """Validate the two Advanced Sampling connections; external-only is enforced."""
    has_sampler = sampler is not None
    has_sigmas = sigmas is not None
    if not has_sampler and not has_sigmas:
        raise ValueError(
            "Motion Director requires both SAMPLER and SIGMAS connections. "
            "Internal sampling was removed: connect KSamplerSelect or "
            "MiniMax-H3 Turbo Sampler together with BasicScheduler or another "
            "ComfyUI scheduler."
        )
    if has_sampler != has_sigmas:
        connected = "SAMPLER" if has_sampler else "SIGMAS"
        missing = "SIGMAS" if has_sampler else "SAMPLER"
        raise ValueError(
            "Motion Director Advanced Sampling is incomplete: %s is connected "
            "but %s is missing. Connect both SAMPLER and SIGMAS."
            % (connected, missing)
        )
    return "external"


def validate_external_sampling(model, sampler, sigmas) -> tuple[torch.Tensor, int]:
    """Fail before loading a model when an Advanced Sampling graph is unsafe."""
    import comfy.model_base
    import comfy.model_sampling

    if sampler is None:
        raise ValueError(
            "Motion Director external sampling requires a SAMPLER connection. "
            "Connect KSamplerSelect or MiniMax-H3 Turbo Sampler."
        )
    if sigmas is None:
        raise ValueError(
            "Motion Director external sampling requires a SIGMAS connection. "
            "Connect BasicScheduler or another ComfyUI scheduler."
        )
    if not isinstance(sigmas, torch.Tensor):
        raise TypeError(
            "Motion Director external sigmas must be ComfyUI SIGMAS (a torch Tensor)."
        )
    if sigmas.ndim != 1:
        raise ValueError(
            "Motion Director external SIGMAS must be one-dimensional; got shape %s."
            % (tuple(sigmas.shape),)
        )
    if int(sigmas.numel()) < 2:
        raise ValueError(
            "Motion Director external SIGMAS needs at least two values "
            "(one sampling step plus its destination)."
        )
    checked = sigmas.detach().float().cpu()
    if not bool(torch.isfinite(checked).all()):
        raise ValueError("Motion Director external SIGMAS contains NaN or infinity.")
    if bool((checked < 0).any()):
        raise ValueError("Motion Director external SIGMAS contains negative values.")
    if bool((checked[1:] > checked[:-1] + 1e-7).any()):
        raise ValueError("Motion Director external SIGMAS must be non-increasing.")

    base_model = getattr(model, "model", None)
    if not isinstance(base_model, comfy.model_base.MiniMaxH3):
        raise ValueError(
            "Motion Director external mode requires a MiniMax H3 MODEL. The "
            "connected MODEL is %s." % type(base_model).__name__
        )
    model_sampling = model.get_model_object("model_sampling")
    if not isinstance(model_sampling, comfy.model_sampling.ModelSamplingAV):
        raise ValueError(
            "Motion Director external mode requires ModelSamplingMiniMaxH3 / "
            "ModelSamplingAV on the connected MODEL. Connect the model through "
            "ModelSamplingMiniMaxH3 before BasicScheduler and Motion Director."
        )
    shift = float(getattr(model_sampling, "shift", 0.0) or 0.0)
    audio_shift = getattr(model_sampling, "audio_shift", None)
    if shift <= 0 or audio_shift is None or float(audio_shift) <= 0:
        raise ValueError(
            "Motion Director external MODEL has incomplete H3 video/audio sigma shifts."
        )
    sigma_max = float(model_sampling.sigma_max.detach().cpu())
    if float(checked.max()) > sigma_max * 1.05 + 1e-6:
        raise ValueError(
            "Motion Director external SIGMAS starts above this H3 model's sigma "
            "range (%.6f > %.6f). Build SIGMAS from the same patched MODEL."
            % (float(checked.max()), sigma_max)
        )
    return checked, int(checked.numel()) - 1


def apply_denoise_to_sigmas(
    sigmas: torch.Tensor, denoise: float
) -> tuple[torch.Tensor, int]:
    """Restrict a connected SIGMAS schedule to its partial-denoise tail.

    ComfyUI core expresses denoise through the schedule: ``KSampler.set_steps``
    builds ``int(steps / denoise)`` sigmas and samples only the last ``steps``
    of them, so the effective denoise is the ratio of executed steps to the full
    schedule.  A connected SIGMAS is already a fixed schedule, so the same
    partial denoise can only be expressed by running its last
    ``round(steps * denoise)`` steps -- the convention ComfyUI's own
    ``SplitSigmasDenoise`` node uses.  ``denoise <= 0`` yields an empty schedule,
    matching core behaviour where the latent is returned unchanged.
    """
    steps = max(int(sigmas.shape[-1]) - 1, 0)
    value = 1.0 if denoise is None else float(denoise)
    if value > 0.9999:
        return sigmas, steps
    if value <= 0.0:
        return sigmas.new_empty(0), 0
    executed = max(1, int(round(steps * value)))
    return sigmas[-(executed + 1):], executed


def sample_single_stage(
    *,
    model,
    positive,
    negative,
    latent,
    seed: int,
    cfg: float,
    # steps / sampler_name / scheduler / shifts are accepted for the frozen
    # legacy executor call contract; internal sampling is gone, so they are
    # ignored. The executed schedule always comes from external SIGMAS.
    steps: int,
    sampler_name: str,
    scheduler: str,
    external_sampler=None,
    external_sigmas=None,
    shift_video: float = 12.0,
    shift_audio: float = 3.0,
    on_phase: PhaseCallback | None = None,
    on_step_preview: StepPreviewCallback | None = None,
    preview_every: int = 1,
    denoise: float = 1.0,
    phase_name: str = "sample",
):
    import comfy.sample
    import comfy.utils

    def notify(phase: str, value: float) -> None:
        if on_phase:
            on_phase(phase, value)

    notify(phase_name, 0)
    sigmas_checked, _full_step_count = validate_external_sampling(
        model, external_sampler, external_sigmas
    )
    # External SIGMAS were built from the caller's already-patched H3 MODEL.
    # Reapplying MiniMaxH3SigmaShift here would clone/double-shift it.
    model_for_sampling = model

    neg = negative if negative else []
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model_for_sampling,
        latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )

    noise = comfy.sample.prepare_noise(
        latent_image,
        int(seed),
        latent.get("batch_index", None),
    )
    noise_mask = latent.get("noise_mask", None)

    every = max(1, int(preview_every))
    model_for_sampling, preview_wrapped = _install_preview_outer_wrapper(
        model_for_sampling, on_step_preview, every
    )

    def callback(step, x0, x, total_steps):
        notify(phase_name, min(1.0, (int(step) + 1) / max(1, int(total_steps))))
        if on_step_preview is not None and not preview_wrapped:
            try:
                if step % every == 0 or step >= max(0, int(total_steps) - 1):
                    on_step_preview(int(step), int(total_steps), x0, None)
            except Exception as exc:
                log.debug("Step preview callback skipped: %s", exc)
        # Intentionally do not install ComfyUI's latent preview callback here.
        # Director owns its bounded side-channel preview; the native
        # sampler preview stays suppressed even when Director Preview is OFF.

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    sigmas_for_sampling, executed_steps = apply_denoise_to_sigmas(sigmas_checked, float(denoise))
    if executed_steps == 0:
        # Same as ComfyUI core: an empty schedule leaves the latent unchanged.
        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = latent_image
        notify(phase_name, 1)
        return out

    try:
        samples = comfy.sample.sample_custom(
            model_for_sampling,
            noise,
            float(cfg),
            external_sampler,
            sigmas_for_sampling,
            positive,
            neg,
            latent_image,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=int(seed),
        )
    except Exception as exc:
        raise RuntimeError(
            "Motion Director external SAMPLER failed while sampling the H3 "
            "nested video/audio latent. Confirm the sampler supports standard "
            "ComfyUI SAMPLER objects and MiniMax H3 NestedTensor inputs. "
            "Original error: %s" % exc
        ) from exc
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    notify(phase_name, 1)
    return out


__all__ = [
    "apply_denoise_to_sigmas",
    "describe_external_sampler",
    "resolve_refine_model",
    "resolve_sampling_mode",
    "sample_single_stage",
    "validate_external_sampling",
]
