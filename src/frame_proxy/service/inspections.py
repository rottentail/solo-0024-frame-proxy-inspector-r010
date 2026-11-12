from __future__ import annotations

from frame_proxy.config import Settings
from frame_proxy.domain.errors import (
    ApplicationError,
    ErrorCode,
    MediaValidationError,
)
from frame_proxy.domain.models import InspectionRecord
from frame_proxy.domain.policies import (
    InspectionRequest,
    validate_clip_id,
    validate_identifier,
)
from frame_proxy.domain.transitions import transition_inspection
from frame_proxy.media.avi import AviDescriptor, probe_avi
from frame_proxy.media.budget import MediaJobGate, WorkingSetBudget
from frame_proxy.media.detector import (
    StreamingCutDetector,
    StreamingSceneAnalyzer,
)
from frame_proxy.media.fingerprint import FrameFingerprint, fingerprint_frame
from frame_proxy.media.frames import decode_frame_at
from frame_proxy.media.source import MediaSource
from frame_proxy.storage.repository import FileRepository


class _FingerprintStream:
    """Lazily re-decode sampled frames and yield one fingerprint at a time.

    Used by the second analysis pass so neither the frames nor their
    fingerprints are materialized up front: each iteration holds exactly one
    source RGB frame and one fingerprint, both dropped on the next step.
    """

    __slots__ = ("_source", "_descriptor", "_sampled_frames", "_budget")

    def __init__(
        self,
        source: MediaSource,
        descriptor: AviDescriptor,
        sampled_frames: tuple[int, ...],
        budget: WorkingSetBudget,
    ) -> None:
        self._source = source
        self._descriptor = descriptor
        self._sampled_frames = sampled_frames
        self._budget = budget

    def __iter__(self):
        for frame_index in self._sampled_frames:
            frame = decode_frame_at(
                self._source,
                self._descriptor,
                frame_index,
                budget=self._budget,
            )
            fingerprint: FrameFingerprint = fingerprint_frame(frame)
            yield fingerprint


class InspectionService:
    def __init__(
        self,
        repository: FileRepository,
        settings: Settings,
        job_gate: MediaJobGate,
    ) -> None:
        self._repository = repository
        self._settings = settings
        self._job_gate = job_gate

    def create(
        self,
        clip_id: str,
        request: InspectionRequest,
    ) -> InspectionRecord:
        normalized_clip_id = validate_clip_id(clip_id)
        clip = self._repository.get_clip(normalized_clip_id)
        if clip.state != "registered":
            raise MediaValidationError(
                "clip is not ready for inspection",
                code=ErrorCode.INVALID_STATE,
            )

        record = InspectionRecord(
            inspection_id=request.inspection_id,
            clip_id=clip.clip_id,
            state="pending",
            sensitivity=request.sensitivity,
            sample_stride=request.sample_stride,
            min_scene_frames=request.min_scene_frames,
        )
        self._repository.create_inspection(record)

        transition_inspection(record, "scanning", reason="frame scan started")
        self._repository.update_inspection(record)
        try:
            with self._job_gate.permit():
                evidence = self._run_scan(clip, record, request)
            record.working_set_evidence = evidence
            return record
        except ApplicationError as exc:
            self._mark_failed(record, exc.code.value, exc.message)
            raise
        except Exception as exc:
            self._mark_failed(record, ErrorCode.INTERNAL.value, str(exc))
            raise ApplicationError(
                ErrorCode.INTERNAL,
                "shot indexing failed",
                status=500,
            ) from exc

    def _run_scan(
        self,
        clip: "object",
        record: InspectionRecord,
        request: InspectionRequest,
    ) -> dict[str, int]:
        budget = WorkingSetBudget(
            f"inspection:{record.inspection_id}",
            frame_byte_size=clip.width * clip.height * 3,
        )
        pin = self._repository.open_clip_media(clip)
        with pin as source:
            descriptor = probe_avi(source)
            if (
                descriptor.width != clip.width
                or descriptor.height != clip.height
                or descriptor.frame_count != clip.frame_count
                or descriptor.frame_offsets != tuple(clip.frame_offsets)
            ):
                raise MediaValidationError(
                    "pinned media geometry disagrees with the registered clip",
                    code=ErrorCode.FRAME_DECODE,
                )
            sampled_frames = tuple(
                range(0, descriptor.frame_count, request.sample_stride)
            )
            # Streaming scan in two passes over the same samples.  Only one
            # RGB frame and one fingerprint are live at a time; the detector
            # keeps just the previous fingerprint plus the running cut list.
            detector = StreamingCutDetector(
                descriptor.frame_count,
                sensitivity=request.sensitivity,
                min_scene_frames=request.min_scene_frames,
            )
            for frame_index in sampled_frames:
                frame = decode_frame_at(
                    source, descriptor, frame_index, budget=budget
                )
                detector.observe(frame_index, fingerprint_frame(frame))
            cuts = detector.finish()

            # Second pass folds each fingerprint into per-scene statistics.
            analyzer = StreamingSceneAnalyzer(
                descriptor.frame_count,
                sampled_frames,
                sensitivity=request.sensitivity,
                min_scene_frames=request.min_scene_frames,
            )
            accumulators_feed = _FingerprintStream(
                source, descriptor, sampled_frames, budget
            )
            scenes = analyzer.build_scenes(cuts, accumulators_feed)

            with pin.verified_commit():
                record.sampled_frames = list(sampled_frames)
                record.cut_frames = list(cuts)
                record.scenes = list(scenes)
                transition_inspection(
                    record, "indexed", reason="scene ranges verified"
                )
                self._repository.update_inspection(record)
        return budget.snapshot()

    def _mark_failed(
        self,
        record: InspectionRecord,
        code: str,
        message: str,
    ) -> None:
        record.error_code = code
        record.error_message = message
        if record.state in {"pending", "scanning"}:
            transition_inspection(record, "failed", reason=code)
        try:
            self._repository.update_inspection(record)
        except Exception:
            pass

    def get(self, inspection_id: str) -> InspectionRecord:
        normalized = validate_identifier(inspection_id, "inspection_id")
        return self._repository.get_inspection(normalized)
