"""Reviewable evidence: per-task working-set and scratch-disk budgets.

Runs both media workflows over a large-frame clip and asserts that, beyond the
exempt index metadata, scene statistics, and final canvas, each task:

* never holds more than two source RGB frames;
* never charges more than 1 MiB of additional resident overhead;
* never uses more than 1 MiB of temporary scratch disk.

It also reports a tracemalloc peak for the whole process so the resident
footprint can be compared against the pre-change baseline.
"""
from __future__ import annotations

import json
import tracemalloc

from frame_proxy.domain.policies import (
    ContactSheetRequest,
    InspectionRequest,
)
from frame_proxy.media.avi_writer import build_sample_avi
from frame_proxy.media.budget import (
    MAX_RESIDENT_FRAMES,
    TASK_OVERHEAD_BYTES,
    TASK_SCRATCH_BYTES,
)
from tools.harness import RunningService, require, success

# A clip whose individual RGB frames are large (640x360 => 691,200 bytes), so
# any leak that retained frames or the whole media would be obvious.
CLIP_WIDTH = 640
CLIP_HEIGHT = 360
SCENE_LENGTHS = (12, 10, 8, 10, 7)
FRAME_RGB_BYTES = CLIP_WIDTH * CLIP_HEIGHT * 3


def main() -> int:
    payload = build_sample_avi(
        width=CLIP_WIDTH,
        height=CLIP_HEIGHT,
        fps=12,
        scene_lengths=SCENE_LENGTHS,
    )
    with RunningService() as service:
        router = service.router
        clip = router._clips.admit("clip-budget", payload)
        require(
            clip.byte_size == len(payload),
            "registered clip size must match the upload",
        )

        tracemalloc.start()
        inspection = router._inspections.create(
            "clip-budget",
            InspectionRequest(
                inspection_id="insp-budget",
                sensitivity=0.55,
                sample_stride=1,
                min_scene_frames=2,
            ),
        )
        _, inspection_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        i_evidence = inspection.working_set_evidence
        require(i_evidence is not None, "inspection must report evidence")

        require(
            i_evidence["peak_resident_frames"] <= MAX_RESIDENT_FRAMES,
            "inspection held more than two source RGB frames",
        )
        require(
            i_evidence["peak_overhead_bytes"] <= TASK_OVERHEAD_BYTES,
            "inspection exceeded its 1 MiB resident overhead budget",
        )
        require(
            i_evidence["frame_byte_size"] == FRAME_RGB_BYTES,
            "frame size used by the ledger must match clip geometry",
        )
        # At task end no source frame remains resident.
        require(
            i_evidence["resident_frames"] == 0,
            "inspection leaked a resident frame after completion",
        )

        tracemalloc.start()
        sheet = router._contact_sheets.create(
            "insp-budget",
            ContactSheetRequest(
                sheet_id="sheet-budget",
                columns=3,
                tile_width=200,
                gutter=6,
                selection="stored",
                background_rgb=(32, 37, 44),
            ),
        )
        _, sheet_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        s_evidence = sheet.working_set_evidence
        require(s_evidence is not None, "contact sheet must report evidence")

        require(
            s_evidence["peak_resident_frames"] <= MAX_RESIDENT_FRAMES,
            "contact sheet held more than two source RGB frames",
        )
        require(
            s_evidence["peak_overhead_bytes"] <= TASK_OVERHEAD_BYTES,
            "contact sheet exceeded its 1 MiB resident overhead budget",
        )
        require(
            s_evidence["peak_scratch_bytes"] <= TASK_SCRATCH_BYTES,
            "contact sheet exceeded its 1 MiB scratch-disk budget",
        )
        require(
            s_evidence["resident_frames"] == 0,
            "contact sheet leaked a resident frame after completion",
        )
        require(
            s_evidence["scratch_bytes"] == 0,
            "contact sheet scratch must be released after commit",
        )

        # No scratch directory or file survives a finished task.
        scratch_root = service.settings.data_root / "scratch"
        if scratch_root.exists():
            leftovers = [str(p) for p in scratch_root.rglob("*")]
        else:
            leftovers = []
        require(leftovers == [], f"scratch leftovers survived: {leftovers}")

    evidence = {
        "clip_bytes": len(payload),
        "frame_rgb_bytes": FRAME_RGB_BYTES,
        "overhead_budget_bytes": TASK_OVERHEAD_BYTES,
        "scratch_budget_bytes": TASK_SCRATCH_BYTES,
        "frame_slot_limit": MAX_RESIDENT_FRAMES,
        "inspection": {
            **i_evidence,
            "process_peak_traced_bytes": inspection_peak,
        },
        "contact_sheet": {
            **s_evidence,
            "process_peak_traced_bytes": sheet_peak,
        },
    }
    success("bounded-working-set", evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
