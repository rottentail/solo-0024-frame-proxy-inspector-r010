from __future__ import annotations

from dataclasses import dataclass

from frame_proxy.domain.errors import ErrorCode, MediaValidationError
from frame_proxy.media.avi import AviDescriptor
from frame_proxy.media.budget import WorkingSetBudget
from frame_proxy.media.source import FileMediaSource, MediaSource


@dataclass(frozen=True, slots=True)
class Frame:
    index: int
    width: int
    height: int
    rgb: bytes


def decode_frame_at(
    source: MediaSource,
    descriptor: AviDescriptor,
    frame_index: int,
    budget: WorkingSetBudget | None = None,
) -> Frame:
    """Decode one frame into RGB using only row-sized transient reads.

    The AVI payload is never materialized: each BGR row is read through the
    media source, converted in place, and appended to the single output RGB
    buffer.  When ``budget`` is supplied the frame occupies one of the task's
    two resident frame slots, and each row's conversion scratch is charged
    against its overhead budget.
    """
    if frame_index < 0 or frame_index >= descriptor.frame_count:
        raise MediaValidationError(
            "frame index is outside the clip",
            code=ErrorCode.MEDIA_BOUNDS,
            details={"frame_index": frame_index},
        )
    offset = descriptor.frame_offsets[frame_index]
    end = offset + descriptor.frame_payload_size
    if end > len(source):
        raise MediaValidationError(
            "frame payload is truncated",
            code=ErrorCode.FRAME_DECODE,
        )

    slot = budget.frame_slot() if budget is not None else _NullContext()
    with slot:
        rgb = bytearray(descriptor.width * descriptor.height * 3)
        row_width = descriptor.width * 3
        for row_index in range(descriptor.height):
            source_row = row_index
            if not descriptor.top_down:
                source_row = descriptor.height - 1 - row_index
            row_start = offset + source_row * descriptor.row_stride
            lease = (
                budget.bytes_lease(descriptor.row_stride)
                if budget is not None
                else _NullContext()
            )
            with lease:
                row = source.read_at(row_start, row_width)
                converted = bytearray(len(row))
                converted[0::3] = row[2::3]
                converted[1::3] = row[1::3]
                converted[2::3] = row[0::3]
            destination = row_index * row_width
            rgb[destination : destination + row_width] = converted
        if isinstance(source, FileMediaSource):
            source.hook_point(frame_index)
        return Frame(
            index=frame_index,
            width=descriptor.width,
            height=descriptor.height,
            rgb=bytes(rgb),
        )


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None
