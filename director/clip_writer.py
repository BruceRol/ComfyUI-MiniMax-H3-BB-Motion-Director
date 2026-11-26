# Portions derived from ComfyUI_MiniMaxH3_Director
# Copyright AIMixer and contributors
# Originally licensed under Apache License 2.0
# Modified for MiniMax H3 Motion Director, 2026-08-09
# This derivative project is distributed under GPL-3.0.
# See NOTICE and LICENSES/Apache-2.0-AIMixer.txt.

"""Per-clip h264 MP4 writer (pyav, single pass).

One writer per clip: video frames are encoded as they are flushed, audio is encoded
from the segment's captured audio dict, and the container is finalized when the clip
is closed. No metadata, no pingpong, no audio-trim remux: the mux is one pass.

Reference pattern: comfy-core ``execution.py`` ``SaveVideo`` (pyav in-process mux).
VideoHelperSuite's h264-mp4 container profile is followed for the colour properties
(bt709) and the AAC audio codec.
"""

from __future__ import annotations

import logging
import os
import re
from fractions import Fraction
from typing import Any

import numpy as np
import torch

try:
    import av
except ImportError:  # pragma: no cover - reported, never silently skipped
    av = None

import folder_paths
from ..lib.image_prep import pad_frames_to_canvas

log = logging.getLogger(__name__)

OUTPUT_SAMPLE_RATE = 44100
AAC_FRAME_SAMPLES = 1024
CONTAINER_OPTIONS = {"movflags": "faststart"}
BT709_COLOR_PROPERTIES = {
    "color_primaries": "bt709",
    "color_transfer": "bt709",
    "colorspace": "bt709",
}
PIX_FMTS = ("yuv420p", "yuv422p", "yuv444p")
PRESET = "medium"


def sanitize_dirname(prefix: str) -> str:
    """Subdirectory name inside the ComfyUI output folder."""
    name = str(prefix or "").strip()
    name = re.sub(r"[^\w.\-]+", "_", name)
    return name or "MiniMaxH3_BB"


def clip_filename(timeline_index: int) -> str:
    """``clip-<prompt number>.mp4`` — numbered from 1, like the timeline card shows.

    ``timeline_index`` is 0-based internally, the editor numbers the prompts from 1,
    so the first clip is ``clip-001.mp4``.
    """
    return f"clip-{max(1, int(timeline_index) + 1):03d}.mp4"


def _time_base(rate: int) -> Fraction:
    """A time_base as a Fraction with a non-zero denominator.

    pyav accepts a ``0/1`` (unset) or ``1/0`` (zero denominator) rational, and
    ``av_rescale_q()`` divides by ``dst.num * src.den``: a zero denominator is an
    integer divide-by-zero inside ffmpeg, which on Windows is a fatal SEH that kills
    the whole ComfyUI process. Every time_base the writer uses is therefore built
    through this helper.
    """
    rate = int(rate)
    if rate <= 0:
        raise ValueError(f"Cannot build a time_base from rate {rate}.")
    return Fraction(1, rate)


def _audio_layout(channels: int) -> str:
    channels = int(channels)
    if channels <= 1:
        return "mono"
    if channels == 2:
        return "stereo"
    raise ValueError(f"Clip writer cannot encode a {channels}-channel audio track.")


def audio_channels(audio: dict[str, Any] | None) -> int:
    """Channel count of an audio dict (``waveform`` is ``[batch, channels, samples]``)."""
    waveform = (audio or {}).get("waveform")
    if waveform is None:
        return 1
    if waveform.ndim >= 3:
        waveform = waveform[0]
    if waveform.ndim < 2:
        return 1
    return max(1, int(waveform.shape[0]))


def silent_audio(
    frame_count: int,
    *,
    fps: float,
    sample_rate: int = OUTPUT_SAMPLE_RATE,
    channels: int = 1,
) -> dict[str, Any]:
    """Silent padded track for a segment that produced no audio."""
    samples = max(1, int(round(int(frame_count) / float(fps or 24.0) * int(sample_rate))))
    return {
        "waveform": torch.zeros((1, max(1, int(channels)), samples), dtype=torch.float32),
        "sample_rate": int(sample_rate),
    }


class ClipWriteResult:
    def __init__(self, path: str, frames: int, audio_samples: int, timeline_index: int = 0) -> None:
        self.path = path
        self.frames = int(frames)
        self.audio_samples = int(audio_samples)
        self.timeline_index = int(timeline_index)


class ClipWriter:
    """One clip's container. Frames and audio are written in export order."""

    def __init__(
        self,
        *,
        timeline_index: int,
        fps: float,
        dirname: str,
        pix_fmt: str,
        crf: int,
        width: int = 0,
        height: int = 0,
        fill: float = 0.5,
        channels: int = 1,
        manager: "ClipWriterManager | None" = None,
    ) -> None:
        self.timeline_index = int(timeline_index)
        self.fps = float(fps or 24.0)
        self.dirname = dirname
        self.pix_fmt = pix_fmt if pix_fmt in PIX_FMTS else "yuv420p"
        self.crf = int(crf)
        # h264 yuv pixel formats need an even canvas.
        self.width = int(width) - (int(width) % 2)
        self.height = int(height) - (int(height) % 2)
        self.fill = float(fill)
        self.channels = max(1, int(channels))
        self.manager = manager
        self.filename = clip_filename(self.timeline_index)
        # PyAV opens an encoder with time_base = 1/framerate for video and
        # 1/sample_rate for audio, so one unit is exactly one frame / one sample.
        self._video_time_base = _time_base(int(round(self.fps)))
        self._audio_time_base = _time_base(OUTPUT_SAMPLE_RATE)
        self._container = None
        self._video_stream = None
        self._audio_stream = None
        self._audio_layout = _audio_layout(self.channels)
        self._resampler = None
        self._frame_count = 0
        self._audio_samples = 0
        self._muxed_any = False
        self._path = ""

    def open(self) -> str:
        if av is None:
            raise RuntimeError("pyav is required for per-clip MP4 output (pip install av).")
        out_dir = os.path.join(folder_paths.get_output_directory(), self.dirname)
        os.makedirs(out_dir, exist_ok=True)
        self._path = os.path.join(out_dir, self.filename)
        log.info("[BB Clip] Writing %s", self._path)
        self._container = av.open(self._path, mode="w")
        self._video_stream = self._container.add_stream("h264", rate=int(round(self.fps)))
        if self.width > 0 and self.height > 0:
            # Fix the container size before the first frame, so a letterboxed clip
            # reports the canvas rather than the first frame's own size.
            self._video_stream.width = self.width
            self._video_stream.height = self.height
        # Pin the time_base on both the codec context and the stream: `add_stream()`
        # leaves it unset (0/1), and a zero numerator/denominator reaches
        # av_rescale_q() as an integer divide-by-zero.
        self._video_stream.codec_context.time_base = self._video_time_base
        self._video_stream.time_base = self._video_time_base
        self._video_stream.pix_fmt = self.pix_fmt
        self._video_stream.options = {
            "crf": str(self.crf),
            "preset": PRESET,
            **BT709_COLOR_PROPERTIES,
        }
        # Every stream must exist before the first packet is muxed: the container header
        # is written on that first mux, and a stream added afterwards has no track and no
        # muxer packet-duration state, which is where the divide-by-zero came from.
        self._ensure_audio_stream(self._audio_layout)
        return self._path

    @property
    def path(self) -> str:
        return self._path

    def encode_frames(self, frames: torch.Tensor) -> list:
        """Encode padded frames and return their packets, without muxing.

        Packets must reach the container in presentation order, so a window can
        encode the region that can no longer change and hold the packets until the
        earlier frames of the clip are final.
        """
        if self._container is None:
            raise RuntimeError("Clip writer is not open.")
        if frames is None:
            raise ValueError(f"Clip writer received no frames for {self.filename}.")
        if frames.ndim == 3:
            # A single frame is the same clip region as a one-frame batch.
            frames = frames.unsqueeze(0)
        if frames.ndim != 4 or int(frames.shape[0]) <= 0:
            raise ValueError(
                f"Clip writer expects a non-empty [F,H,W,C] tensor, got {tuple(frames.shape)} "
                f"for {self.filename}."
            )
        if int(frames.shape[3]) != 3:
            raise ValueError(
                f"Clip writer needs 3 channels for rgb24, got {tuple(frames.shape)} "
                f"for {self.filename}."
            )
        # The plan canvas is authoritative; without one the frame size is the canvas.
        canvas_w = self.width or int(frames.shape[2])
        canvas_h = self.height or int(frames.shape[1])
        canvas_w -= canvas_w % 2
        canvas_h -= canvas_h % 2
        padded = pad_frames_to_canvas(frames, canvas_w, canvas_h, fill=self.fill)
        array = padded.detach().cpu().float().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
        stream = self._video_stream
        packets = []
        for row in array:
            row = np.ascontiguousarray(row)
            if row.ndim != 3 or int(row.shape[2]) != 3:
                raise ValueError(
                    f"Clip writer produced a {row.ndim}-D frame {tuple(row.shape)} from "
                    f"{tuple(frames.shape)} for {self.filename}."
                )
            frame = av.VideoFrame.from_ndarray(row, format="rgb24")
            frame = frame.reformat(format=self.pix_fmt)
            # CodecContext.encode() only copies the frame's pts/duration onto the
            # packet, so synthetic frames must be timed by the caller.
            frame.pts = self._frame_count
            frame.duration = 1
            frame.time_base = self._video_time_base
            packets.extend(stream.encode(frame))
            self._frame_count += 1
        return packets

    def write_packet(self, packet) -> None:
        if self._container is None:
            raise RuntimeError("Clip writer is not open.")
        self._mux_packets(packet)

    def write_frames(self, frames: torch.Tensor) -> int:
        self._mux_packets(self.encode_frames(frames))
        return self._frame_count

    def write_audio(self, audio: dict[str, Any] | None) -> int:
        if self._container is None:
            raise RuntimeError("Clip writer is not open.")
        if not audio:
            return self._audio_samples
        waveform = audio.get("waveform")
        if waveform is None:
            return self._audio_samples
        if waveform.ndim >= 3:
            waveform = waveform[0]
        if waveform.ndim < 2:
            return self._audio_samples
        waveform = waveform.detach().cpu().float()
        # After the batch axis is dropped the shape is [channels, samples].
        channels = int(waveform.shape[0])
        samples = int(waveform.shape[1])
        if channels <= 0 or samples <= 0:
            return self._audio_samples
        layout = _audio_layout(channels)
        source_rate = int(audio.get("sample_rate") or OUTPUT_SAMPLE_RATE)
        if source_rate <= 0:
            source_rate = OUTPUT_SAMPLE_RATE
        stream = self._ensure_audio_stream(layout)
        resampler = None
        if source_rate != OUTPUT_SAMPLE_RATE:
            resampler = av.AudioResampler(format="fltp", layout=layout, rate=OUTPUT_SAMPLE_RATE)
        source_time_base = _time_base(source_rate)
        step = AAC_FRAME_SAMPLES
        try:
            for start in range(0, samples, step):
                block = waveform[:, start : start + step]
                if int(block.shape[1]) <= 0:
                    continue
                frame = av.AudioFrame.from_ndarray(
                    np.ascontiguousarray(block.numpy()), format="fltp", layout=layout
                )
                frame.sample_rate = source_rate if resampler is not None else OUTPUT_SAMPLE_RATE
                if resampler is not None:
                    # The filter graph takes its time_base from the first frame; an unset
                    # one makes ffmpeg guess, so pin it to 1/source_rate.
                    frame.time_base = source_time_base
                    for out in resampler.resample(frame):
                        self._encode_audio_frame(out, stream)
                else:
                    self._encode_audio_frame(frame, stream)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                f"Clip writer failed to encode audio for {self.filename} "
                f"({channels} channel(s), {samples} samples at {source_rate} Hz): {exc}"
            ) from exc
        return self._audio_samples

    def _ensure_audio_stream(self, layout: str):
        if self._audio_stream is None:
            stream = self._container.add_stream("aac", rate=OUTPUT_SAMPLE_RATE)
            stream.layout = layout
            # `add_stream()` leaves a new stream with an unset time_base (0/1), so
            # pin both the codec context and the stream before the first frame.
            stream.codec_context.time_base = self._audio_time_base
            stream.time_base = self._audio_time_base
            self._audio_stream = stream
            self._audio_layout = layout
        elif layout != self._audio_layout:
            # The layout is read when the header is written, which happens on the first
            # mux, so it can only change while the container is still empty.
            if self._muxed_any:
                raise ValueError(
                    f"Clip writer cannot change {self.filename} to a {layout} audio track: "
                    f"the container header was already written with {self._audio_layout}."
                )
            self._audio_stream.layout = layout
            self._audio_layout = layout
        return self._audio_stream

    def _check_time_base(self, stream, what: str) -> None:
        # A rational with a zero numerator or denominator reaches av_rescale_q() as an
        # integer divide-by-zero, which on Windows is a fatal SEH that kills ComfyUI.
        for label, value in (("codec_context", stream.codec_context.time_base), ("stream", stream.time_base)):
            num, den = int(value.num), int(value.den)
            if num <= 0 or den <= 0:
                raise ValueError(
                    f"Clip writer refuses to mux {self.filename}: {what} {label} time_base "
                    f"is {num}/{den}."
                )

    def _check_packet(self, packet) -> None:
        num, den = int(packet.time_base.num), int(packet.time_base.den)
        if num <= 0 or den <= 0:
            raise ValueError(
                f"Clip writer refuses to mux {self.filename}: a packet carries the time_base "
                f"{num}/{den}."
            )

    def _mux_packets(self, packets) -> None:
        # encode() returns a list in pyav >= 15 and a single Packet in older versions,
        # so a caller can hand over a list of lists; flatten one level before muxing.
        if packets is None:
            return
        if not isinstance(packets, (list, tuple)):
            packets = [packets]
        for packet in packets:
            if packet is None:
                continue
            if isinstance(packet, (list, tuple)):
                self._mux_packets(packet)
                continue
            stream = packet.stream
            if stream is None:
                stream = self._audio_stream if packet.stream_index == 1 else self._video_stream
            if stream is None:
                raise ValueError(f"Clip writer cannot mux a packet for {self.filename} without a stream.")
            self._check_time_base(stream, "audio" if stream is self._audio_stream else "video")
            self._check_packet(packet)
            try:
                self._container.mux(packet)
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(
                    f"Clip writer failed to mux a packet for {self.filename}: {exc}"
                ) from exc
            self._muxed_any = True

    def _encode_audio_frame(self, frame, stream) -> None:
        samples = int(frame.samples)
        if samples <= 0:
            return
        # pts is the sample offset: the audio encoder time_base is 1/sample_rate.
        frame.pts = self._audio_samples
        frame.time_base = self._audio_time_base
        self._mux_packets(stream.encode(frame))
        self._audio_samples += samples

    def close(self) -> ClipWriteResult:
        if self._container is None:
            raise RuntimeError("Clip writer is not open.")
        if self._video_stream is not None:
            self._mux_packets(self._video_stream.encode())
        if self._audio_stream is not None:
            self._mux_packets(self._audio_stream.encode())
        self._container.close()
        log.info(
            "[BB Clip] Wrote %s (%d frames, %d audio samples)",
            self._path,
            self._frame_count,
            self._audio_samples,
        )
        result = ClipWriteResult(
            self._path, self._frame_count, self._audio_samples, self.timeline_index
        )
        if self.manager is not None:
            self.manager.results.append(result)
        return result


class ClipWriterManager:
    """One writer per clip, keyed by timeline index."""

    def __init__(
        self,
        *,
        node_id: str = "",
        dirname_prefix: str,
        fps: float,
        pix_fmt: str,
        crf: int,
        width: int = 0,
        height: int = 0,
    ) -> None:
        self.node_id = str(node_id or "")
        self.dirname = sanitize_dirname(dirname_prefix)
        self.fps = float(fps or 24.0)
        self.pix_fmt = pix_fmt if pix_fmt in PIX_FMTS else "yuv420p"
        self.crf = int(crf)
        # h264 yuv pixel formats need an even canvas.
        self.width = int(width) - (int(width) % 2)
        self.height = int(height) - (int(height) % 2)
        self._writers: dict[int, ClipWriter] = {}
        self.results: list[ClipWriteResult] = []

    def writer(self, timeline_index: int, channels: int = 1) -> ClipWriter:
        index = int(timeline_index)
        writer = self._writers.get(index)
        if writer is None:
            writer = ClipWriter(
                timeline_index=index,
                fps=self.fps,
                dirname=self.dirname,
                pix_fmt=self.pix_fmt,
                crf=self.crf,
                width=self.width,
                height=self.height,
                channels=max(1, int(channels)),
                manager=self,
            )
            writer.open()
            self._writers[index] = writer
        return writer

    def summary_lines(self) -> list[str]:
        lines = [f"Status: SUCCESS ({len(self.results)} clip(s)) -> {self.dirname}/"]
        for result in sorted(self.results, key=lambda item: int(item.timeline_index)):
            lines.append(f"{clip_filename(result.timeline_index)} ({result.frames} frames)")
        return lines


__all__ = [
    "ClipWriter",
    "ClipWriterManager",
    "ClipWriteResult",
    "PIX_FMTS",
    "OUTPUT_SAMPLE_RATE",
    "clip_filename",
    "sanitize_dirname",
    "audio_channels",
]
