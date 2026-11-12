"""Reviewable evidence: source replacement can never produce a mixed artifact.

For each media workflow this tool mutates the registered clip file while the
task is in flight, using the production frame-access diagnostic hook so the
mutation lands after some frames have already been read:

* in-place content change (same inode, different bytes);
* rename replacement (different inode behind the same registered path).

In every tamper case the task must fail with a stable media error, must not
publish a successful artifact, and must leave a failed job record.  An
untampered control run on the same inputs must still succeed and match the
committed digest.
"""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

from frame_proxy.domain.errors import ApplicationError
from frame_proxy.domain.policies import (
    ContactSheetRequest,
    InspectionRequest,
)
from frame_proxy.media.avi_writer import build_sample_avi
from frame_proxy.media.source import register_frame_access_hook
from tools.harness import RunningService, require, require_status, success

SCENE_LENGTHS = (12, 10, 8, 10, 7)


def _replace_bytes(path: Path, payload: bytes) -> None:
    fd = os.open(os.fspath(path), os.O_WRONLY | os.O_TRUNC)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def _replace_file(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".replacement")
    temporary.write_bytes(payload)
    os.replace(os.fspath(temporary), os.fspath(path))


def _wait_for_frame(clip_id: str, target_frame: int) -> threading.Event:
    event = threading.Event()

    def hook(_clip_id: str, frame_index: int) -> None:
        if frame_index >= target_frame and not event.is_set():
            event.set()

    register_frame_access_hook(clip_id, hook)
    return event


def _inspection_failed(service: RunningService, inspection_id: str) -> None:
    response = service.request("GET", f"/v1/inspections/{inspection_id}")
    require_status(response, 200, "inspection lookup after tamper")
    body = response.json()["inspection"]
    require(body["state"] == "failed", "tampered inspection must be failed")
    require(
        body["error_code"] == "frame_decode_failed",
        f"unexpected tamper error code: {body['error_code']}",
    )


def _sheet_failed(service: RunningService, sheet_id: str) -> None:
    response = service.request("GET", f"/v1/contact-sheets/{sheet_id}")
    require_status(response, 200, "contact sheet lookup after tamper")
    body = response.json()["contact_sheet"]
    require(body["state"] == "failed", "tampered contact sheet must be failed")
    missing = service.request(
        "GET", f"/v1/contact-sheets/{sheet_id}/artifact"
    )
    require_status(missing, 409, "no artifact may be published for a failed sheet")


def run() -> int:
    original = build_sample_avi(
        width=320, height=180, fps=12, scene_lengths=SCENE_LENGTHS
    )
    # Different complete footage (different colors via longer scenes geometry
    # is not possible with the same writer, so build a distinct payload by
    # using different scene lengths -> different frame count and bytes).
    replacement = build_sample_avi(
        width=320,
        height=180,
        fps=12,
        scene_lengths=(8, 8, 8, 8, 7, 8),
    )
    require(
        hashlib.sha256(original).hexdigest()
        != hashlib.sha256(replacement).hexdigest(),
        "tamper payload must differ from the registered payload",
    )

    with RunningService() as service:
        router = service.router
        data_root = service.settings.data_root

        # ---- Control: untampered inspection and sheet succeed. ----------
        control_clip = router._clips.admit("clip-control", original)
        control_insp = router._inspections.create(
            "clip-control",
            InspectionRequest(
                inspection_id="insp-control",
                sensitivity=0.55,
                sample_stride=1,
                min_scene_frames=2,
            ),
        )
        require(
            control_insp.state == "indexed",
            "control inspection must succeed",
        )
        control_sheet = router._contact_sheets.create(
            "insp-control",
            ContactSheetRequest(
                sheet_id="sheet-control",
                columns=3,
                tile_width=96,
                gutter=4,
                selection="stored",
                background_rgb=(32, 37, 44),
            ),
        )
        require(
            control_sheet.state == "ready",
            "control contact sheet must succeed",
        )

        cases = (
            ("inplace", _replace_bytes),
            ("rename", _replace_file),
        )

        for mode, mutate in cases:
            clip_id = f"clip-{mode}"
            inspection_id = f"insp-{mode}"
            sheet_id = f"sheet-{mode}"
            router._clips.admit(clip_id, original)
            media_path = data_root / "clips" / clip_id / "clip.avi"

            # Inspection tamper: mutate once several frames were scanned.
            event = _wait_for_frame(clip_id, target_frame=6)
            tamper_error: list[Exception] = []

            def tamper() -> None:
                event.wait(timeout=10)
                try:
                    mutate(media_path, replacement)
                except Exception as exc:  # pragma: no cover - diagnostic only
                    tamper_error.append(exc)

            worker = threading.Thread(target=tamper, daemon=True)
            worker.start()
            try:
                router._inspections.create(
                    clip_id,
                    InspectionRequest(
                        inspection_id=inspection_id,
                        sensitivity=0.55,
                        sample_stride=1,
                        min_scene_frames=2,
                    ),
                )
                raised = None
            except ApplicationError as exc:
                raised = exc
            worker.join(timeout=10)
            register_frame_access_hook(clip_id, None)
            require(not tamper_error, f"{mode} tamper thread failed")
            require(
                raised is not None and raised.status == 422,
                f"{mode} inspection must reject tampered media (422)",
            )
            _inspection_failed(service, inspection_id)

            # Contact sheet tamper on a fresh clip: mutate mid-composition.
            sheet_clip_id = f"clip-sheet-{mode}"
            sheet_inspection_id = f"insp-sheet-{mode}"
            router._clips.admit(sheet_clip_id, original)
            good_insp = router._inspections.create(
                sheet_clip_id,
                InspectionRequest(
                    inspection_id=sheet_inspection_id,
                    sensitivity=0.55,
                    sample_stride=1,
                    min_scene_frames=2,
                ),
            )
            require(
                good_insp.state == "indexed",
                "pre-tamper inspection must be indexed",
            )
            sheet_media_path = (
                data_root / "clips" / sheet_clip_id / "clip.avi"
            )
            event = _wait_for_frame(sheet_clip_id, target_frame=1)
            worker = threading.Thread(
                target=lambda: (event.wait(timeout=10),
                                mutate(sheet_media_path, replacement)),
                daemon=True,
            )
            worker.start()
            try:
                router._contact_sheets.create(
                    sheet_inspection_id,
                    ContactSheetRequest(
                        sheet_id=sheet_id,
                        columns=3,
                        tile_width=96,
                        gutter=4,
                        selection="stored",
                        background_rgb=(32, 37, 44),
                    ),
                )
                raised = None
            except ApplicationError as exc:
                raised = exc
            worker.join(timeout=10)
            register_frame_access_hook(sheet_clip_id, None)
            require(
                raised is not None and raised.status == 422,
                f"{mode} contact sheet must reject tampered media (422)",
            )
            _sheet_failed(service, sheet_id)

        # The control artifact remains fetchable and digest-consistent.
        artifact = router._contact_sheets.get_artifact("sheet-control")
        require(
            hashlib.sha256(artifact).hexdigest()
            == control_sheet.artifact_digest,
            "untampered control artifact digest must remain stable",
        )

    success(
        "source-tamper-rejection",
        {
            "modes": [mode for mode, _ in cases],
            "rejection_status": 422,
            "control_digest": control_sheet.artifact_digest,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
