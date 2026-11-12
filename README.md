# Frame Proxy Inspector

Frame Proxy Inspector is a small Python backend for turning uncompressed AVI
footage into two review artifacts:

- a deterministic shot index with contiguous scene ranges;
- a bounded PPM contact sheet with one tile per indexed scene.

The service verifies the RIFF/AVI container, frame geometry, and frame chunk
count before it admits a clip. Shot indexing then computes luma and color
fingerprints from sampled frames and applies a minimum-scene guard to hard
cuts. Contact sheet requests are checked against a fixed pixel budget before
frame decoding starts.

Automated unit, integration, and browser tests are intentionally deferred to a
later task stage. This baseline ships executable workflow checks instead.

## Requirements

- Python 3.11 or newer
- No third-party packages
- No external media service

## Start the service

```bash
export PYTHONPATH=src
python3 -m frame_proxy serve --host 127.0.0.1 --port 8765
```

The service prints its bound URL. Press `Ctrl+C` to stop it.

## Generate demonstration footage

```bash
PYTHONPATH=src python3 -m frame_proxy demo-avi \
  --output demo/scene-walk.avi \
  --width 128 \
  --height 72 \
  --fps 12
```

The generated file is an uncompressed 24-bit AVI with four color scenes and
17 total frames. It is suitable for exercising all three workflows.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | service check |
| `PUT` | `/v1/clips/{clip_id}` | admit an `application/x-avi` payload |
| `GET` | `/v1/clips/{clip_id}` | read clip geometry and digest |
| `POST` | `/v1/clips/{clip_id}/inspections` | index shot boundaries |
| `GET` | `/v1/inspections/{inspection_id}` | read an immutable inspection |
| `POST` | `/v1/inspections/{inspection_id}/contact-sheets` | compose a PPM grid |
| `GET` | `/v1/contact-sheets/{sheet_id}` | read the tile map and digest |
| `GET` | `/v1/contact-sheets/{sheet_id}/artifact` | fetch the PPM bytes |

Identifiers use lowercase letters, digits, `_`, and `-`, begin with a letter,
and contain 3 to 48 characters.

Example inspection request:

```json
{
  "inspection_id": "inspection-demo-a",
  "sensitivity": 0.55,
  "sample_stride": 1,
  "min_scene_frames": 2
}
```

Example contact sheet request:

```json
{
  "sheet_id": "sheet-demo-a",
  "columns": 3,
  "tile_width": 96,
  "gutter": 4,
  "selection": "stored",
  "background": "#20252c"
}
```

`selection` may be `stored`, `midpoint`, or `first`. The rendered PPM is
binary P6 with a maximum value of 255.

## Data layout

`FRAME_PROXY_DATA_DIR` selects the data root. Its default is
`.frame-proxy-data` in the current working directory.

```text
.frame-proxy-data/
├── clips/<clip_id>/
│   ├── clip.avi
│   └── record.json
├── inspections/<inspection_id>.json
├── contact-sheets/<sheet_id>/
│   ├── record.json
│   └── sheet.ppm
├── scratch/                 # bounded per-task temp area; empty after commit
└── locks/
```

Records are written through temporary files and renamed into place. Contact
sheet bytes and their ready record are verified before the record becomes
visible.

## Source layout

```text
src/frame_proxy/
├── domain/      records, request rules, and state transitions
├── http/        request routing and response mapping
├── media/       streaming AVI access, budgets, fingerprints, and PPM grids
├── service/     workflow orchestration and failure handling
└── storage/     namespaced files, locks, and atomic writes

tools/
├── harness.py
├── check_clip_intake.py
├── check_shot_index.py
├── check_contact_sheet.py
├── check_bounded_memory.py
├── check_source_tamper.py
└── check_controlled_concurrency.py
```

## Build and workflow checks

Build:

```bash
python3 -m compileall -q src tools
```

Run one workflow check at a time:

```bash
PYTHONPATH=src:. python3 tools/check_clip_intake.py
PYTHONPATH=src:. python3 tools/check_shot_index.py
PYTHONPATH=src:. python3 tools/check_contact_sheet.py
```

Each check starts a real loopback HTTP server, generates media through the
production AVI writer, completes the named workflow, and verifies persisted
state or artifact bytes. Temporary records are removed when the check exits.

## Bounded media working set

Shot indexing and contact sheet composition never load a clip into memory.
They read the registered AVI through a streaming, identity-pinned media
source. Beyond the index metadata, scene statistics, and final canvas, each
media task is independently limited to:

- **1 MiB** of additional resident Python heap, tracked by a per-task
  `WorkingSetBudget`;
- **at most two source RGB frames** (both workflows hold just one at a time);
- **1 MiB** of temporary scratch disk, tracked by a per-task `TaskScratch`.

A process-wide `MediaJobGate` bounds how many media tasks run at once
(default 2, override with `FRAME_PROXY_MAX_MEDIA_JOBS`), so the aggregate
media footprint is at most `limit × (1 MiB + two RGB frames)`.

The pin hashes the full clip on open and again immediately before commit, and
also re-checks the inode identity of the registered path. If the source file
is replaced or rewritten while a task is in flight, the task fails with a
stable `frame_decode_failed` (HTTP 422), records a terminal failed job, and
never publishes a mixed-content artifact.

Reviewable evidence checks (each prints its measured ledger):

```bash
PYTHONPATH=src:. python3 tools/check_bounded_memory.py
PYTHONPATH=src:. python3 tools/check_source_tamper.py
PYTHONPATH=src:. python3 tools/check_controlled_concurrency.py
```

- `check_bounded_memory` asserts the frame/overhead/scratch limits on a
  large-frame clip and reports the tracemalloc peak.
- `check_source_tamper` mutates the source in place and by rename mid-task and
  confirms both workflows reject it while an untampered control still
  succeeds.
- `check_controlled_concurrency` saturates the job gate and confirms the
  active count never exceeds the limit while every job yields identical
  results.


## Environment variables

| Name | Default | Purpose |
| --- | --- | --- |
| `FRAME_PROXY_DATA_DIR` | `.frame-proxy-data` | durable data root |
| `FRAME_PROXY_MAX_MEDIA_JOBS` | `2` | max concurrent media processing tasks (1–16) |

Clip, upload, JSON, frame dimension, frame count, and contact sheet limits are
declared in `src/frame_proxy/config.py`. The per-task 1 MiB overhead/scratch
caps and the two-frame limit are declared in
`src/frame_proxy/media/budget.py`.

## Input boundary

The current adapter accepts one uncompressed 24-bit BGR video stream in a
RIFF/AVI container. Audio streams, compressed video, and additional container
formats are intentionally outside this baseline. The extension seams for those
capabilities are recorded in `.task-source.json`.

