"""Per-task working-set budgets and controlled media concurrency.

Each media processing task (shot indexing or contact sheet composition) may
keep resident, beyond its index metadata, scene statistics and final canvas:

* at most :data:`TASK_OVERHEAD_BYTES` (1 MiB) of additional Python heap
  allocations tracked by :class:`WorkingSetBudget`; and
* at most two source RGB frames, obtained explicitly through the budget and
  released back to it.

Temporary scratch files written by a task are bounded by the same 1 MiB cap
through :class:`TaskScratch`.  These limits are per task, so bounded
concurrency multiplies only by the number of permits handed out by
:class:`MediaJobGate`.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Iterator

from frame_proxy.domain.errors import ErrorCode, MediaValidationError

#: Additional resident bytes allowed per task outside the exempt objects.
TASK_OVERHEAD_BYTES = 1 * 1024 * 1024

#: Maximum source RGB frames resident per task at any moment.
MAX_RESIDENT_FRAMES = 2

#: Maximum temporary scratch bytes per task (same cap as resident overhead).
TASK_SCRATCH_BYTES = 1 * 1024 * 1024


class WorkingSetBudget:
    """Thread-safe resident working-set ledger for one task."""

    __slots__ = (
        "label",
        "frame_byte_size",
        "_frames",
        "_overhead",
        "_peak_overhead",
        "_peak_frames",
        "_lock",
    )

    def __init__(self, label: str, frame_byte_size: int) -> None:
        if frame_byte_size <= 0:
            raise ValueError("frame byte size must be positive")
        self.label = label
        self.frame_byte_size = frame_byte_size
        self._frames = 0
        self._overhead = 0
        self._peak_overhead = 0
        self._peak_frames = 0
        self._lock = threading.Lock()

    @property
    def resident_frames(self) -> int:
        return self._frames

    @property
    def overhead_bytes(self) -> int:
        return self._overhead

    @property
    def peak_overhead_bytes(self) -> int:
        return self._peak_overhead

    @property
    def peak_resident_frames(self) -> int:
        return self._peak_frames

    @property
    def peak_frame_bytes(self) -> int:
        return self._peak_frames * self.frame_byte_size

    def acquire_bytes(self, size: int) -> None:
        if size < 0:
            raise ValueError("allocation size must not be negative")
        with self._lock:
            projected = self._overhead + size
            if projected > TASK_OVERHEAD_BYTES:
                raise MediaValidationError(
                    "task working set exceeded its 1 MiB overhead budget",
                    code=ErrorCode.FRAME_DECODE,
                    details={
                        "task": self.label,
                        "requested_bytes": size,
                        "resident_bytes": self._overhead,
                        "budget_bytes": TASK_OVERHEAD_BYTES,
                    },
                )
            self._overhead = projected
            if projected > self._peak_overhead:
                self._peak_overhead = projected

    def release_bytes(self, size: int) -> None:
        with self._lock:
            if size > self._overhead:
                raise ValueError("released more bytes than were acquired")
            self._overhead -= size

    @contextmanager
    def bytes_lease(self, size: int) -> Iterator[None]:
        self.acquire_bytes(size)
        try:
            yield
        finally:
            self.release_bytes(size)

    def acquire_frame_slot(self) -> None:
        with self._lock:
            if self._frames >= MAX_RESIDENT_FRAMES:
                raise MediaValidationError(
                    "task working set may hold at most two source RGB frames",
                    code=ErrorCode.FRAME_DECODE,
                    details={
                        "task": self.label,
                        "resident_frames": self._frames,
                        "limit": MAX_RESIDENT_FRAMES,
                    },
                )
            self._frames += 1
            if self._frames > self._peak_frames:
                self._peak_frames = self._frames

    def release_frame_slot(self) -> None:
        with self._lock:
            if self._frames <= 0:
                raise ValueError("no frame slot to release")
            self._frames -= 1

    @contextmanager
    def frame_slot(self) -> Iterator[None]:
        self.acquire_frame_slot()
        try:
            yield
        finally:
            self.release_frame_slot()

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "overhead_bytes": self._overhead,
                "peak_overhead_bytes": self._peak_overhead,
                "resident_frames": self._frames,
                "peak_resident_frames": self._peak_frames,
                "frame_byte_size": self.frame_byte_size,
                "peak_frame_bytes": self._peak_frames * self.frame_byte_size,
                "overhead_budget_bytes": TASK_OVERHEAD_BYTES,
                "frame_slot_limit": MAX_RESIDENT_FRAMES,
            }


class TaskScratch:
    """Bounded per-task temporary disk area inside the data root."""

    __slots__ = ("directory", "_written", "_peak", "_lock")

    def __init__(self, root: Path, task_name: str) -> None:
        self.directory = root / "scratch" / task_name
        self._written = 0
        self._peak = 0
        self._lock = threading.Lock()

    @property
    def written_bytes(self) -> int:
        return self._written

    @property
    def peak_bytes(self) -> int:
        return self._peak

    def __enter__(self) -> "TaskScratch":
        self.directory.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.cleanup()

    def reserve(self, size: int) -> None:
        if size < 0:
            raise ValueError("scratch size must not be negative")
        with self._lock:
            projected = self._written + size
            if projected > TASK_SCRATCH_BYTES:
                raise MediaValidationError(
                    "task scratch exceeded its 1 MiB disk budget",
                    code=ErrorCode.STORAGE,
                    details={
                        "requested_bytes": size,
                        "reserved_bytes": self._written,
                        "budget_bytes": TASK_SCRATCH_BYTES,
                    },
                )
            self._written = projected
            if projected > self._peak:
                self._peak = projected

    def release(self, size: int) -> None:
        with self._lock:
            if size > self._written:
                raise ValueError("released more scratch than was reserved")
            self._written -= size

    def write_file(self, name: str, payload: bytes) -> Path:
        """Write a bounded temporary file; callers remove it after use."""
        self.reserve(len(payload))
        target = self.directory / name
        try:
            fd = os.open(
                os.fspath(target),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                os.write(fd, payload)
                os.fsync(fd)
            finally:
                os.close(fd)
        except BaseException:
            self.release(len(payload))
            try:
                target.unlink()
            except FileNotFoundError:
                pass
            raise
        return target

    def remove_file(self, path: Path) -> None:
        size = path.stat().st_size
        path.unlink()
        self.release(size)

    def reclaim_file(self, size: int) -> None:
        """Release the reservation for a scratch file published elsewhere.

        The file has been atomically renamed out of the scratch directory, so
        its bytes no longer occupy task scratch even though this task still
        owns the budget reservation.  The size is passed explicitly because
        the scratch path no longer exists after the rename.
        """
        self.release(size)

    def cleanup(self) -> None:
        if not self.directory.exists():
            return
        for child in self.directory.iterdir():
            try:
                if child.is_file():
                    child.unlink()
            except FileNotFoundError:
                pass
        try:
            self.directory.rmdir()
        except OSError:
            pass
        with self._lock:
            self._written = 0

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "scratch_bytes": self._written,
                "peak_scratch_bytes": self._peak,
                "scratch_budget_bytes": TASK_SCRATCH_BYTES,
            }


class MediaJobGate:
    """Process-wide semaphore bounding concurrent media processing tasks."""

    __slots__ = ("limit", "_semaphore", "_active", "_peak", "_lock", "_total")

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("media job concurrency must be at least one")
        self.limit = limit
        self._semaphore = threading.BoundedSemaphore(limit)
        self._active = 0
        self._peak = 0
        self._total = 0
        self._lock = threading.Lock()

    @contextmanager
    def permit(self) -> Iterator[None]:
        self._semaphore.acquire()
        with self._lock:
            self._active += 1
            self._total += 1
            if self._active > self._peak:
                self._peak = self._active
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
            self._semaphore.release()

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "concurrency_limit": self.limit,
                "active_jobs": self._active,
                "peak_jobs": self._peak,
                "total_jobs": self._total,
            }
