from __future__ import annotations

from frame_proxy.config import Settings
from frame_proxy.domain.errors import (
    ApplicationError,
    ErrorCode,
    MediaValidationError,
    StateError,
)
from frame_proxy.domain.models import ContactSheetRecord
from frame_proxy.domain.policies import ContactSheetRequest, validate_identifier
from frame_proxy.domain.transitions import transition_contact_sheet
from frame_proxy.media.avi import probe_avi
from frame_proxy.media.budget import (
    MediaJobGate,
    TaskScratch,
    WorkingSetBudget,
)
from frame_proxy.media.contact import compose_grid, plan_grid
from frame_proxy.media.frames import Frame, decode_frame_at
from frame_proxy.media.ppm import parse_ppm_header
from frame_proxy.storage.repository import FileRepository


class ContactSheetService:
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
        inspection_id: str,
        request: ContactSheetRequest,
    ) -> ContactSheetRecord:
        normalized_id = validate_identifier(inspection_id, "inspection_id")
        inspection = self._repository.get_inspection(normalized_id)
        if inspection.state != "indexed":
            raise StateError(
                "contact sheets require an indexed inspection",
                details={"state": inspection.state},
            )
        clip = self._repository.get_clip(inspection.clip_id)
        plan = plan_grid(
            scene_count=len(inspection.scenes),
            source_width=clip.width,
            source_height=clip.height,
            request=request,
            max_scenes=self._settings.max_sheet_scenes,
            max_pixels=self._settings.max_sheet_pixels,
        )

        record = ContactSheetRecord(
            sheet_id=request.sheet_id,
            inspection_id=inspection.inspection_id,
            state="planned",
            columns=request.columns,
            tile_width=request.tile_width,
            tile_height=plan.tile_height,
            gutter=request.gutter,
            background_rgb=request.background_rgb,
            selection=request.selection,
            rows=plan.rows,
            image_width=plan.image_width,
            image_height=plan.image_height,
            artifact_name="sheet.ppm",
            artifact_digest=None,
            tile_map=[],
        )
        self._repository.create_contact_sheet(record)
        transition_contact_sheet(record, "compositing", reason="layout reserved")
        self._repository.update_contact_sheet(record)

        try:
            with self._job_gate.permit():
                ready_record, evidence = self._compose(
                    clip, inspection, request, plan, record
                )
            ready_record.working_set_evidence = evidence
            return ready_record
        except ApplicationError as exc:
            self._mark_failed(record, exc.code.value, exc.message)
            raise
        except Exception as exc:
            self._mark_failed(record, ErrorCode.INTERNAL.value, str(exc))
            raise ApplicationError(
                ErrorCode.INTERNAL,
                "contact sheet composition failed",
                status=500,
            ) from exc

    def _compose(
        self,
        clip: "object",
        inspection: "object",
        request: ContactSheetRequest,
        plan: "object",
        record: ContactSheetRecord,
    ) -> tuple[ContactSheetRecord, dict[str, int]]:
        budget = WorkingSetBudget(
            f"contact-sheet:{record.sheet_id}",
            frame_byte_size=clip.width * clip.height * 3,
        )
        scratch = TaskScratch(
            self._repository.paths.root,
            f"contact-sheet-{record.sheet_id}",
        )
        pin = self._repository.open_clip_media(clip)
        with scratch, pin as source:
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
            scenes = tuple(inspection.scenes)

            def frame_provider(frame_index: int) -> Frame:
                # One source frame is live at a time: the frame returned here
                # is tiled and dropped before the next provider call.
                return decode_frame_at(
                    source,
                    descriptor,
                    frame_index,
                    budget=budget,
                )

            result = compose_grid(
                plan=plan,
                scenes=scenes,
                frame_provider=frame_provider,
                background_rgb=request.background_rgb,
                selection=request.selection,
                budget=budget,
            )
            # The PPM artifact bytes are transiently resident next to the
            # exempt canvas, so charge them while they are held and spooled.
            with budget.bytes_lease(len(result.payload)):
                header = parse_ppm_header(result.payload)
                if (
                    header.width != record.image_width
                    or header.height != record.image_height
                ):
                    raise ValueError(
                        "composed PPM dimensions disagree with the plan"
                    )

                ready_record = ContactSheetRecord.from_dict(record.as_dict())
                ready_record.artifact_digest = result.digest
                ready_record.tile_map = list(result.tile_map)
                transition_contact_sheet(
                    ready_record,
                    "ready",
                    reason="artifact verified",
                )

                # Re-prove the source is still the registered clip before any
                # artifact becomes visible, then spool through bounded scratch.
                with pin.verified_commit():
                    artifact_size = len(result.payload)
                    scratch_path = scratch.write_file(
                        f".{record.sheet_id}.ppm.tmp",
                        result.payload,
                    )
                    self._repository.commit_contact_sheet_from_scratch(
                        ready_record,
                        scratch_path,
                    )
                    # The file was renamed into the artifact location; its
                    # bytes no longer count against this task's scratch budget.
                    scratch.reclaim_file(artifact_size)
            evidence = {**budget.snapshot(), **scratch.snapshot()}
            return ready_record, evidence

    def _mark_failed(
        self,
        record: ContactSheetRecord,
        code: str,
        message: str,
    ) -> None:
        record.error_code = code
        record.error_message = message
        if record.state in {"planned", "compositing"}:
            transition_contact_sheet(record, "failed", reason=code)
        try:
            self._repository.update_contact_sheet(record)
        except Exception:
            pass

    def get(self, sheet_id: str) -> ContactSheetRecord:
        normalized = validate_identifier(sheet_id, "sheet_id")
        return self._repository.get_contact_sheet(normalized)

    def get_artifact(self, sheet_id: str) -> bytes:
        normalized = validate_identifier(sheet_id, "sheet_id")
        record = self._repository.get_contact_sheet(normalized)
        if record.state != "ready":
            raise StateError(
                "contact sheet is not ready",
                details={"state": record.state},
            )
        return self._repository.get_contact_sheet_artifact(
            record.sheet_id,
            record.artifact_name,
        )
