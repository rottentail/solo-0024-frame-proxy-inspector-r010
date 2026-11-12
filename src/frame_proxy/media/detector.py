from __future__ import annotations

from dataclasses import dataclass

from frame_proxy.domain.errors import ErrorCode, MediaValidationError
from frame_proxy.domain.models import SceneRecord
from frame_proxy.media.fingerprint import FrameFingerprint, fingerprint_distance


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    sampled_frames: tuple[int, ...]
    cut_frames: tuple[int, ...]
    scenes: tuple[SceneRecord, ...]


def _threshold(sensitivity: float) -> float:
    return 0.50 - sensitivity * 0.30


def _detect_cut_frames(
    frame_count: int,
    sampled_frames: tuple[int, ...],
    fingerprints: tuple[FrameFingerprint, ...],
    sensitivity: float,
    min_scene_frames: int,
) -> list[int]:
    if len(sampled_frames) != len(fingerprints) or not sampled_frames:
        raise MediaValidationError(
            "sampled frames and fingerprints must be non-empty and aligned",
            code=ErrorCode.FRAME_DECODE,
        )
    threshold = _threshold(sensitivity)
    cuts: list[int] = []
    last_cut = 0
    for index in range(1, len(sampled_frames)):
        distance = fingerprint_distance(fingerprints[index - 1], fingerprints[index])
        candidate = sampled_frames[index]
        if distance < threshold:
            continue
        if candidate - last_cut < min_scene_frames:
            continue
        if frame_count - candidate < min_scene_frames:
            continue
        cuts.append(candidate)
        last_cut = candidate
    return cuts


def _scene_records(
    frame_count: int,
    sampled_frames: tuple[int, ...],
    fingerprints: tuple[FrameFingerprint, ...],
    cuts: list[int],
) -> tuple[SceneRecord, ...]:
    starts = [0, *cuts]
    ends = [*cuts, frame_count]
    fingerprint_by_frame = dict(zip(sampled_frames, fingerprints))
    scenes: list[SceneRecord] = []

    for scene_index, (start, end) in enumerate(zip(starts, ends)):
        in_scene = [
            (frame_index, fingerprint_by_frame[frame_index])
            for frame_index in sampled_frames
            if start <= frame_index < end
        ]
        midpoint = start + (end - start) // 2
        if in_scene:
            representative_frame, _ = min(
                in_scene,
                key=lambda item: abs(item[0] - midpoint),
            )
            count = len(in_scene)
            mean_luma = sum(item[1].mean_luma for item in in_scene) / count
            contrast = sum(item[1].contrast for item in in_scene) / count
            dominant = (
                round(sum(item[1].mean_red for item in in_scene) / count),
                round(sum(item[1].mean_green for item in in_scene) / count),
                round(sum(item[1].mean_blue for item in in_scene) / count),
            )
        else:
            representative_frame = midpoint
            nearest_index = min(
                range(len(sampled_frames)),
                key=lambda index: abs(sampled_frames[index] - midpoint),
            )
            fingerprint = fingerprints[nearest_index]
            mean_luma = fingerprint.mean_luma
            contrast = fingerprint.contrast
            dominant = (
                round(fingerprint.mean_red),
                round(fingerprint.mean_green),
                round(fingerprint.mean_blue),
            )
        scenes.append(
            SceneRecord(
                index=scene_index,
                start_frame=start,
                end_frame=end,
                representative_frame=representative_frame,
                mean_luma=mean_luma,
                contrast=contrast,
                dominant_rgb=dominant,
            )
        )
    return tuple(scenes)


def analyze_frames(
    frame_count: int,
    sampled_frames: tuple[int, ...],
    fingerprints: tuple[FrameFingerprint, ...],
    *,
    sensitivity: float,
    min_scene_frames: int,
) -> AnalysisResult:
    cuts = _detect_cut_frames(
        frame_count,
        sampled_frames,
        fingerprints,
        sensitivity,
        min_scene_frames,
    )
    scenes = _scene_records(frame_count, sampled_frames, fingerprints, cuts)
    _require_contiguous(frame_count, scenes)
    return AnalysisResult(
        sampled_frames=sampled_frames,
        cut_frames=tuple(cuts),
        scenes=scenes,
    )


def _require_contiguous(
    frame_count: int, scenes: tuple[SceneRecord, ...]
) -> None:
    if not scenes or scenes[0].start_frame != 0:
        raise MediaValidationError(
            "scene ranges must begin at frame zero",
            code=ErrorCode.FRAME_DECODE,
        )
    if scenes[-1].end_frame != frame_count:
        raise MediaValidationError(
            "scene ranges must end at the clip frame count",
            code=ErrorCode.FRAME_DECODE,
        )
    for left, right in zip(scenes, scenes[1:]):
        if left.end_frame != right.start_frame:
            raise MediaValidationError(
                "scene ranges must be contiguous",
                code=ErrorCode.FRAME_DECODE,
            )


class StreamingCutDetector:
    """First pass: detect hard cuts holding only two fingerprints.

    Feed sampled ``(frame_index, fingerprint)`` pairs in order; the detector
    keeps just the previous fingerprint and the running cut list, so its
    resident state is independent of the clip's frame count.
    """

    __slots__ = ("frame_count", "threshold", "min_scene_frames",
                 "_previous_fp", "_last_cut", "cuts", "_seen")

    def __init__(
        self,
        frame_count: int,
        *,
        sensitivity: float,
        min_scene_frames: int,
    ) -> None:
        self.frame_count = frame_count
        self.threshold = _threshold(sensitivity)
        self.min_scene_frames = min_scene_frames
        self._previous_fp: FrameFingerprint | None = None
        self._last_cut = 0
        self.cuts: list[int] = []
        self._seen = 0

    def observe(self, frame_index: int, fingerprint: FrameFingerprint) -> None:
        self._seen += 1
        if self._previous_fp is not None:
            distance = fingerprint_distance(self._previous_fp, fingerprint)
            candidate = frame_index
            if (
                distance >= self.threshold
                and candidate - self._last_cut >= self.min_scene_frames
                and self.frame_count - candidate >= self.min_scene_frames
            ):
                self.cuts.append(candidate)
                self._last_cut = candidate
        self._previous_fp = fingerprint

    def finish(self) -> list[int]:
        if self._seen == 0:
            raise MediaValidationError(
                "sampled frames and fingerprints must be non-empty and aligned",
                code=ErrorCode.FRAME_DECODE,
            )
        return self.cuts


@dataclass(slots=True)
class _SceneAccumulator:
    start: int
    end: int
    midpoint: int
    count: int = 0
    luma_sum: float = 0.0
    contrast_sum: float = 0.0
    red_sum: float = 0.0
    green_sum: float = 0.0
    blue_sum: float = 0.0
    representative_frame: int | None = None
    representative_distance: int | None = None
    # Used only for the empty-scene fallback to the globally nearest sample.
    nearest_frame: int | None = None
    nearest_distance: int | None = None
    nearest_fp: FrameFingerprint | None = None


class StreamingSceneAnalyzer:
    """Two-pass analyzer whose resident state is bounded per scene.

    Pass one (:class:`StreamingCutDetector`) finds cuts with two fingerprints
    live.  Pass two streams the same sampled fingerprints again and folds each
    into one accumulator per final scene.  Only the accumulator set (the
    exempt scene statistics) and one fingerprint at a time are resident,
    instead of one fingerprint per clip frame.
    """

    __slots__ = ("frame_count", "sampled_frames")

    def __init__(
        self,
        frame_count: int,
        sampled_frames: tuple[int, ...],
        *,
        sensitivity: float,
        min_scene_frames: int,
    ) -> None:
        if not sampled_frames:
            raise MediaValidationError(
                "sampled frames and fingerprints must be non-empty and aligned",
                code=ErrorCode.FRAME_DECODE,
            )
        # Sensitivity and minimum scene length are retained in the signature
        # for symmetry with the original analyzer but only affect cut
        # detection, which is handled by StreamingCutDetector.
        self.frame_count = frame_count
        self.sampled_frames = sampled_frames

    def build_scenes(
        self,
        cuts: list[int],
        fingerprints: "object",
    ) -> tuple[SceneRecord, ...]:
        starts = [0, *cuts]
        ends = [*cuts, self.frame_count]
        accumulators = [
            _SceneAccumulator(
                start=start,
                end=end,
                midpoint=start + (end - start) // 2,
            )
            for start, end in zip(starts, ends)
        ]

        for frame_index, fingerprint in zip(self.sampled_frames, fingerprints):
            scene_index = self._scene_index_for(cuts, frame_index)
            target = accumulators[scene_index]
            target.count += 1
            target.luma_sum += fingerprint.mean_luma
            target.contrast_sum += fingerprint.contrast
            target.red_sum += fingerprint.mean_red
            target.green_sum += fingerprint.mean_green
            target.blue_sum += fingerprint.mean_blue
            distance = abs(frame_index - target.midpoint)
            if (
                target.representative_distance is None
                or distance < target.representative_distance
            ):
                target.representative_distance = distance
                target.representative_frame = frame_index

            # Globally nearest sampled frame for any scene that ends up empty.
            for accumulator in accumulators:
                nearest = abs(frame_index - accumulator.midpoint)
                if (
                    accumulator.nearest_distance is None
                    or nearest < accumulator.nearest_distance
                ):
                    accumulator.nearest_distance = nearest
                    accumulator.nearest_frame = frame_index
                    accumulator.nearest_fp = fingerprint

        scenes: list[SceneRecord] = []
        for scene_index, accumulator in enumerate(accumulators):
            if accumulator.count > 0:
                count = accumulator.count
                representative_frame = accumulator.representative_frame
                mean_luma = accumulator.luma_sum / count
                contrast = accumulator.contrast_sum / count
                dominant = (
                    round(accumulator.red_sum / count),
                    round(accumulator.green_sum / count),
                    round(accumulator.blue_sum / count),
                )
            else:
                fingerprint = accumulator.nearest_fp
                representative_frame = accumulator.midpoint
                mean_luma = fingerprint.mean_luma
                contrast = fingerprint.contrast
                dominant = (
                    round(fingerprint.mean_red),
                    round(fingerprint.mean_green),
                    round(fingerprint.mean_blue),
                )
            scenes.append(
                SceneRecord(
                    index=scene_index,
                    start_frame=accumulator.start,
                    end_frame=accumulator.end,
                    representative_frame=representative_frame,
                    mean_luma=mean_luma,
                    contrast=contrast,
                    dominant_rgb=dominant,
                )
            )
        result = tuple(scenes)
        _require_contiguous(self.frame_count, result)
        return result

    def _scene_index_for(self, cuts: list[int], frame_index: int) -> int:
        # Scenes are ordered and contiguous; binary search the cut boundary.
        low, high = 0, len(cuts)
        while low < high:
            mid = (low + high) // 2
            if cuts[mid] <= frame_index:
                low = mid + 1
            else:
                high = mid
        return low
