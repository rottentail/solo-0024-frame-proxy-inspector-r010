"""Streaming media access with source identity pinning.

The shot indexing and contact sheet workflows must not hold an entire AVI in
memory.  A :class:`MediaSource` exposes the random-access byte reads that AVI
probing and frame decoding need without materializing the payload, while a
:class:`PinnedClipMedia` keeps the registered clip open for the lifetime of a
task and proves, with a full digest before and after processing, that every
successful artifact still belongs to the registered clip bytes.

If the clip file is replaced or rewritten while a task is running, the pinned
descriptor either keeps reading the original inode (rename/replace) and the
final digest mismatch rejects the task, or observes the changed bytes in place
and the final digest mismatch rejects it.  A task can therefore never combine
an inspection scope from one clip with frames from another into a successful
artifact.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from types import TracebackType
from typing import Iterator, Protocol, runtime_checkable

from frame_proxy.domain.errors import ErrorCode, MediaValidationError

#: Diagnostic seam used by the workflow evidence checks to mutate the pinned
#: media while a task is in flight.  Production code never registers a hook.
FrameAccessObserver = "callable[[str, int], None]"
_FRAME_ACCESS_HOOKS: dict[str, "FrameAccessObserver"] = {}


def register_frame_access_hook(
    clip_id: str,
    hook: "FrameAccessObserver | None",
) -> None:
    """Register or remove a diagnostic per-frame hook for ``clip_id``."""
    if hook is None:
        _FRAME_ACCESS_HOOKS.pop(clip_id, None)
    else:
        _FRAME_ACCESS_HOOKS[clip_id] = hook


def _run_hook(clip_id: str, frame_index: int) -> None:
    hook = _FRAME_ACCESS_HOOKS.get(clip_id)
    if hook is not None:
        hook(clip_id, frame_index)


@runtime_checkable
class MediaSource(Protocol):
    """Random-access byte reader over AVI media."""

    def __len__(self) -> int: ...

    def read_at(self, offset: int, size: int) -> bytes: ...

    def uint32_at(self, offset: int) -> int: ...


class MemorySource:
    """Media source over an already materialized byte buffer (clip intake)."""

    __slots__ = ("_buffer",)

    def __init__(self, buffer: bytes) -> None:
        self._buffer = buffer

    def __len__(self) -> int:
        return len(self._buffer)

    def read_at(self, offset: int, size: int) -> bytes:
        return self._buffer[offset : offset + size]

    def uint32_at(self, offset: int) -> int:
        return int.from_bytes(self._buffer[offset : offset + 4], "little")


class FileMediaSource:
    """Streaming media source backed by an open file descriptor.

    Reads use :func:`os.pread`, so the descriptor position never moves and the
    same source can serve concurrent bounded tasks.  Only the requested slices
    are ever resident.
    """

    __slots__ = ("_fd", "_size", "_clip_id")

    def __init__(self, fd: int, size: int, clip_id: str) -> None:
        self._fd = fd
        self._size = size
        self._clip_id = clip_id

    def __len__(self) -> int:
        return self._size

    def read_at(self, offset: int, size: int) -> bytes:
        if size < 0 or offset < 0 or offset + size > self._size:
            raise MediaValidationError(
                "AVI read extends beyond the pinned media",
                code=ErrorCode.FRAME_DECODE,
            )
        if size == 0:
            return b""
        chunks: list[bytes] = []
        remaining = size
        cursor = offset
        while remaining:
            chunk = os.pread(self._fd, remaining, cursor)
            if not chunk:
                raise MediaValidationError(
                    "pinned media ended before the requested read",
                    code=ErrorCode.FRAME_DECODE,
                )
            chunks.append(chunk)
            cursor += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def uint32_at(self, offset: int) -> int:
        return int.from_bytes(self.read_at(offset, 4), "little")

    def hook_point(self, frame_index: int) -> None:
        """Diagnostic seam invoked once per decoded frame."""
        _run_hook(self._clip_id, frame_index)

    def streaming_digest(self) -> str:
        """Hash the complete pinned media without holding it all at once."""
        hasher = hashlib.sha256()
        remaining = self._size
        cursor = 0
        block = 128 * 1024
        while remaining:
            chunk = os.pread(self._fd, min(block, remaining), cursor)
            if not chunk:
                raise MediaValidationError(
                    "pinned media ended while computing its digest",
                    code=ErrorCode.FRAME_DECODE,
                )
            hasher.update(chunk)
            cursor += len(chunk)
            remaining -= len(chunk)
        return hasher.hexdigest()


class PinnedClipMedia:
    """Open, identity-checked media handle scoped to one processing task.

    The clip is pinned at open with its registered digest and byte size.  A
    full streaming digest is taken before processing and again before commit;
    both must equal the registered digest, and the inode identity captured at
    open must not change.  Any mismatch fails the task with a stable media
    error instead of publishing a mixed-content artifact.
    """

    __slots__ = (
        "_path",
        "_clip_id",
        "_expected_digest",
        "_expected_size",
        "_fd",
        "_stat_identity",
        "_source",
    )

    def __init__(
        self,
        path: "os.PathLike[str] | str",
        *,
        clip_id: str,
        expected_digest: str,
        expected_size: int,
    ) -> None:
        self._path = os.fspath(path)
        self._clip_id = clip_id
        self._expected_digest = expected_digest
        self._expected_size = expected_size
        self._fd: int | None = None
        self._stat_identity: tuple[int, int] | None = None
        self._source: FileMediaSource | None = None

    def __enter__(self) -> FileMediaSource:
        fd = os.open(self._path, os.O_RDONLY)
        try:
            stat = os.fstat(fd)
            if stat.st_size != self._expected_size:
                raise MediaValidationError(
                    "registered clip size does not match the stored media",
                    code=ErrorCode.FRAME_DECODE,
                    details={
                        "registered_size": self._expected_size,
                        "stored_size": stat.st_size,
                    },
                )
            source = FileMediaSource(fd, stat.st_size, self._clip_id)
            digest = source.streaming_digest()
            if digest != self._expected_digest:
                raise MediaValidationError(
                    "stored clip digest does not match its registered record",
                    code=ErrorCode.FRAME_DECODE,
                )
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        self._stat_identity = (stat.st_dev, stat.st_ino)
        self._source = source
        return source

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
            self._source = None

    @contextmanager
    def verified_commit(self) -> Iterator[FileMediaSource]:
        """Re-verify identity immediately before a task commits its result."""
        if self._source is None or self._fd is None:
            raise MediaValidationError(
                "clip media is not pinned",
                code=ErrorCode.FRAME_DECODE,
            )
        stat = os.fstat(self._fd)
        identity = (stat.st_dev, stat.st_ino)
        if identity != self._stat_identity or stat.st_size != self._expected_size:
            raise MediaValidationError(
                "clip media changed identity during processing",
                code=ErrorCode.FRAME_DECODE,
            )
        # The open descriptor alone is not enough: a rename replacement keeps
        # the old inode readable behind this fd while the registered path now
        # points at different bytes.  Re-stat the path and require the same
        # inode identity.
        try:
            path_stat = os.stat(self._path)
        except FileNotFoundError as exc:
            raise MediaValidationError(
                "registered clip media was removed during processing",
                code=ErrorCode.FRAME_DECODE,
            ) from exc
        if (path_stat.st_dev, path_stat.st_ino) != self._stat_identity:
            raise MediaValidationError(
                "clip media was replaced during processing; refusing mixed artifact",
                code=ErrorCode.FRAME_DECODE,
            )
        digest = self._source.streaming_digest()
        if digest != self._expected_digest:
            raise MediaValidationError(
                "clip media changed during processing; refusing mixed artifact",
                code=ErrorCode.FRAME_DECODE,
            )
        yield self._source


def as_media_source(buffer: bytes) -> MemorySource:
    """Wrap an in-memory upload buffer as a media source."""
    return MemorySource(buffer)
