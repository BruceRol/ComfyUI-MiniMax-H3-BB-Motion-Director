from director.low_ram import (
    HEAD_WINDOW_FRAMES,
    MAX_CONTEXT_FRAMES,
    TAIL_WINDOW_FRAMES,
    context_tail_frames,
    tail_window_frames,
    window_sizes,
)


def test_context_tail_is_sized_from_the_widget_and_not_from_a_constant():
    # The retained context region is the region select_context_span reads, so it follows
    # the configured context_length; MAX_CONTEXT_FRAMES is only the widget maximum.
    assert context_tail_frames(5) == 5
    assert context_tail_frames(22) == 22
    assert context_tail_frames(MAX_CONTEXT_FRAMES) == MAX_CONTEXT_FRAMES
    assert context_tail_frames(100) == MAX_CONTEXT_FRAMES


def test_tail_window_widens_only_when_the_context_pipeline_is_active():
    assert tail_window_frames(requested_context=22, context_pipeline_active=False) == TAIL_WINDOW_FRAMES
    assert tail_window_frames(requested_context=22, context_pipeline_active=True) == 22
    assert tail_window_frames(requested_context=39, context_pipeline_active=True) == 39


def test_window_sizes_never_exceed_the_chunk():
    assert window_sizes(60, requested_context=22, context_pipeline_active=True) == (12, 22)
    assert window_sizes(4, requested_context=22, context_pipeline_active=True) == (4, 4)


def test_context_region_reaching_into_the_head_keeps_the_whole_chunk():
    # head + tail > chunk: ClipWindow falls back to the whole chunk, so the window sizes
    # must not shrink the context region below the configured length.
    head, tail = window_sizes(30, requested_context=39, context_pipeline_active=True)
    assert head == HEAD_WINDOW_FRAMES
    assert tail == 30
    assert head + tail > 30


def test_low_ram_on_and_off_cover_the_same_context_span():
    # The consumer decides its span from the segment's true length; the retained tail must
    # cover that span in both modes, so the tail is never smaller than the span the
    # configured context_length can produce.
    for chunk_frames in (60, 40, 30, 20, 10):
        for requested in (1, 5, 22, 39):
            _head, tail = window_sizes(
                chunk_frames,
                requested_context=requested,
                context_pipeline_active=True,
            )
            span = min(context_tail_frames(requested), chunk_frames)
            assert tail >= span
