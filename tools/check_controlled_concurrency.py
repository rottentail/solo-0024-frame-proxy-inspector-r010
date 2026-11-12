"""Reviewable evidence: bounded media concurrency and per-task isolation.

Fires more media jobs than the configured concurrency limit and proves that:

* the active job count never exceeds the limit (a process-wide gate);
* every admitted job still completes successfully with identical results;
* each task independently stays inside its frame/overhead/scratch budgets;
* total resident media is bounded by ``limit * (1 MiB + two RGB frames)``.
"""
from __future__ import annotations

import json
import threading

from frame_proxy.domain.policies import (
    ContactSheetRequest,
    InspectionRequest,
)
from frame_proxy.media.avi_writer import build_sample_avi
from tools.harness import RunningService, require, success

CONCURRENCY_LIMIT = 2
JOB_COUNT = 5


def main() -> int:
    payload = build_sample_avi(
        width=320, height=180, fps=12, scene_lengths=(12, 10, 8, 10, 7)
    )
    with RunningService() as service:
        router = service.router
        settings = service.settings
        require(
            settings.max_media_jobs == CONCURRENCY_LIMIT,
            "service must be configured with the expected media job limit",
        )
        gate = router._inspections._job_gate
        require(gate.limit == CONCURRENCY_LIMIT, "gate limit mismatch")

        clip = router._clips.admit("clip-concurrent", payload)
        frame_rgb_bytes = clip.width * clip.height * 3

        # A barrier releases all worker threads at once, so jobs genuinely
        # contend for the permits instead of running sequentially.
        barrier = threading.Barrier(JOB_COUNT)
        results: list[dict] = []
        errors: list[Exception] = []
        results_lock = threading.Lock()

        def worker(job_index: int) -> None:
            try:
                barrier.wait(timeout=10)
                inspection = router._inspections.create(
                    "clip-concurrent",
                    InspectionRequest(
                        inspection_id=f"insp-conc-{job_index}",
                        sensitivity=0.55,
                        sample_stride=1,
                        min_scene_frames=2,
                    ),
                )
                sheet = router._contact_sheets.create(
                    inspection.inspection_id,
                    ContactSheetRequest(
                        sheet_id=f"sheet-conc-{job_index}",
                        columns=3,
                        tile_width=96,
                        gutter=4,
                        selection="stored",
                        background_rgb=(32, 37, 44),
                    ),
                )
                with results_lock:
                    results.append(
                        {
                            "job": job_index,
                            "cuts": list(inspection.cut_frames),
                            "inspection_evidence": inspection.working_set_evidence,
                            "sheet_evidence": sheet.working_set_evidence,
                            "sheet_digest": sheet.artifact_digest,
                        }
                    )
            except Exception as exc:  # pragma: no cover - surfaced below
                with results_lock:
                    errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(JOB_COUNT)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        require(not errors, f"concurrent jobs failed: {errors}")
        require(len(results) == JOB_COUNT, "every concurrent job must finish")
        gate_snapshot = gate.snapshot()
        require(
            gate_snapshot["peak_jobs"] <= CONCURRENCY_LIMIT,
            "active media jobs exceeded the concurrency limit",
        )
        require(
            gate_snapshot["peak_jobs"] == CONCURRENCY_LIMIT,
            "the workload must actually saturate the gate",
        )
        require(
            gate_snapshot["total_jobs"] == 2 * JOB_COUNT,
            "each task (inspection + sheet) must acquire a permit",
        )

        # All jobs compute identical deterministic results.
        cut_sets = {tuple(item["cuts"]) for item in results}
        require(len(cut_sets) == 1, "concurrent inspections must agree")
        digests = {item["sheet_digest"] for item in results}
        require(len(digests) == 1, "concurrent contact sheets must agree")

        # Per-task budgets hold under contention.
        for item in results:
            for key in ("inspection_evidence", "sheet_evidence"):
                evidence = item[key]
                require(
                    evidence["peak_resident_frames"] <= 2,
                    f"{key} exceeded two resident frames",
                )
                require(
                    evidence["peak_overhead_bytes"] <= 1024 * 1024,
                    f"{key} exceeded 1 MiB overhead",
                )
            require(
                item["sheet_evidence"]["peak_scratch_bytes"] <= 1024 * 1024,
                "sheet scratch exceeded 1 MiB",
            )

        worst_overhead = max(
            max(
                item["inspection_evidence"]["peak_overhead_bytes"],
                item["sheet_evidence"]["peak_overhead_bytes"],
            )
            for item in results
        )
        bounded_resident = (
            CONCURRENCY_LIMIT * (1024 * 1024 + 2 * frame_rgb_bytes)
        )
        evidence = {
            "jobs_submitted": JOB_COUNT,
            "concurrency_limit": CONCURRENCY_LIMIT,
            "gate": gate_snapshot,
            "frame_rgb_bytes": frame_rgb_bytes,
            "worst_task_overhead_bytes": worst_overhead,
            "max_media_resident_bytes": bounded_resident,
            "shared_sheet_digest": next(iter(digests)),
            "shared_cuts": [list(value) for value in sorted(cut_sets)],
        }
        success("controlled-media-concurrency", evidence)
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
