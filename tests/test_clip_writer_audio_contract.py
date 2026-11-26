from pathlib import Path


def _source() -> str:
    return Path("director/clip_writer.py").read_text(encoding="utf-8")


def test_clip_files_are_numbered_from_one():
    # The editor numbers the prompts from 1, so the first clip must be clip-001.mp4.
    source = _source()
    body = source.split("def clip_filename", 1)[1].split("def _time_base", 1)[0]
    assert "max(1, int(timeline_index) + 1)" in body
    assert "max(0, int(timeline_index))" not in source


def test_time_base_helper_rejects_a_zero_rate():
    # av_rescale_q() divides by dst.num * src.den, so a 0/1 or 1/0 time_base is an
    # integer divide-by-zero inside ffmpeg (a fatal SEH on Windows, killing ComfyUI).
    source = _source()
    helper = source.split("def _time_base", 1)[1].split("def _audio_layout", 1)[0]
    assert "if rate <= 0:" in helper
    assert "Fraction(1, rate)" in helper
    assert "_time_base(int(round(self.fps)))" in source
    assert "_time_base(OUTPUT_SAMPLE_RATE)" in source


def test_audio_layout_uses_the_channel_axis_not_the_sample_axis():
    # After the batch axis is dropped the waveform is [channels, samples]; reading
    # shape[1] as the channel count made every mono track fail.
    source = _source()
    assert "def _audio_layout(channels: int) -> str:" in source
    body = source.split("def write_audio", 1)[1].split("def _ensure_audio_stream", 1)[0]
    assert "channels = int(waveform.shape[0])" in body
    assert "samples = int(waveform.shape[1])" in body
    assert "if channels <= 0 or samples <= 0:" in body
    assert "_audio_layout(channels)" in body


def test_audio_stream_time_base_is_pinned_before_the_first_frame():
    source = _source()
    stream_block = source.split("def _ensure_audio_stream", 1)[1].split("def _encode_audio_frame", 1)[0]
    assert "stream.codec_context.time_base = self._audio_time_base" in stream_block
    assert "stream.time_base = self._audio_time_base" in stream_block

    open_block = source.split("def open(self)", 1)[1].split("def encode_frames", 1)[0]
    assert "self._video_stream.codec_context.time_base = self._video_time_base" in open_block
    assert "self._video_stream.time_base = self._video_time_base" in open_block


def test_audio_frames_are_guarded_and_muxed_one_packet_at_a_time():
    source = _source()
    encode_block = source.split("def _encode_audio_frame", 1)[1].split("def close", 1)[0]
    assert "if samples <= 0:" in encode_block
    assert "frame.time_base = self._audio_time_base" in encode_block
    assert "self._mux_packets(stream.encode(frame))" in encode_block
    assert "self._container.mux(stream.encode(frame))" not in source

    resampler_branch = source.split("if resampler is not None:", 1)[1]
    assert "frame.time_base = source_time_base" in resampler_branch


def test_audio_failures_name_the_clip_being_written():
    source = _source()
    body = source.split("def write_audio", 1)[1].split("def _ensure_audio_stream", 1)[0]
    assert "Clip writer failed to encode audio for {self.filename}" in body


def test_every_stream_is_configured_before_the_first_mux():
    # The container header is written on the first mux. A stream added afterwards has no
    # muxer track and no priv_pts denominator, and frac_add() divides by that denominator
    # without guarding it: an integer divide-by-zero, a fatal SEH on Windows.
    source = _source()
    open_block = source.split("def open(self)", 1)[1].split("def encode_frames", 1)[0]
    assert "self._ensure_audio_stream(self._audio_layout)" in open_block
    mux_block = source.split("def _mux_packets", 1)[1].split("def close", 1)[0]
    assert "self._check_time_base(stream," in mux_block
    assert "self._check_packet(packet)" in mux_block
    assert "self._muxed_any = True" in mux_block
    assert "num <= 0 or den <= 0" in source
    assert "Clip writer failed to mux a packet for {self.filename}" in mux_block
    write_block = source.split("def write_packet", 1)[1].split("def write_frames", 1)[0]
    assert "self._mux_packets(packet)" in write_block


def test_encode_frames_returns_a_flat_packet_list():
    # CodecContext.encode() returns a list in pyav >= 15, so appending it made
    # encode_frames() hand a list of lists to the muxer and packet.stream failed.
    source = _source()
    body = source.split("def encode_frames", 1)[1].split("def write_packet", 1)[0]
    assert "packets.extend(stream.encode(frame))" in body
    assert "packets.append(stream.encode(frame))" not in source
    mux_block = source.split("def _mux_packets", 1)[1].split("def _encode_audio_frame", 1)[0]
    assert "if isinstance(packet, (list, tuple)):" in mux_block
    assert "self._mux_packets(packet)" in mux_block


def test_the_audio_layout_can_only_change_while_the_container_is_empty():
    source = _source()
    body = source.split("def _ensure_audio_stream", 1)[1].split("def _check_time_base", 1)[0]
    assert "elif layout != self._audio_layout:" in body
    assert "if self._muxed_any:" in body
    assert "the container header was already written with {self._audio_layout}" in body
