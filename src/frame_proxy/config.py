from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: Default bound on concurrently resident media processing tasks.
DEFAULT_MAX_MEDIA_JOBS = 2


@dataclass(frozen=True, slots=True)
class Settings:
    data_root: Path
    max_clip_bytes: int = 64 * 1024 * 1024
    max_upload_bytes: int = 64 * 1024 * 1024
    max_json_bytes: int = 128 * 1024
    max_frame_width: int = 1920
    max_frame_height: int = 1080
    max_frame_count: int = 3000
    max_sheet_pixels: int = 250_000
    max_sheet_scenes: int = 36
    request_timeout_seconds: float = 30.0
    #: Maximum simultaneously resident media processing tasks.  Each task is
    #: independently bounded to 1 MiB overhead plus two source RGB frames.
    max_media_jobs: int = DEFAULT_MAX_MEDIA_JOBS

    @classmethod
    def from_env(cls) -> "Settings":
        data_root = Path(
            os.environ.get("FRAME_PROXY_DATA_DIR", ".frame-proxy-data")
        ).expanduser()
        max_media_jobs = DEFAULT_MAX_MEDIA_JOBS
        raw_jobs = os.environ.get("FRAME_PROXY_MAX_MEDIA_JOBS")
        if raw_jobs:
            try:
                max_media_jobs = max(1, min(16, int(raw_jobs)))
            except ValueError:
                max_media_jobs = DEFAULT_MAX_MEDIA_JOBS
        return cls(data_root=data_root, max_media_jobs=max_media_jobs)

    def ensure_ready(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
