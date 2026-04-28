from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from буревестник.настройки import (
    MAX_TRACKED_OBJECTS,
    TRACK_MAX_DISTANCE,
    TRACK_MAX_LOST_FRAMES,
    TRACK_MEMORY_DISTANCE,
    TRACK_MEMORY_FRAMES,
    TRACK_MIN_HITS,
    TRACK_MIN_IOU,
)
from буревестник.сущности import Detection, Track


@dataclass
class TrackMemory:
    """Короткая память о пропавшем объекте, чтобы не выдавать ему новый ID рядом с тем же местом."""

    track_id: int
    bbox: tuple[int, int, int, int]
    label: str
    confidence: float
    center: tuple[int, int]
    hits: int
    trace: tuple[tuple[int, int], ...]
    age_frames: int = 0


class SimpleTracker:
    """Простой трекер по центрам рамок.

    Это еще не финальный Deep SORT/ByteTrack. Зато он не требует лишних
    библиотек, не падает из-за KalmanFilter и дает главное для прототипа:
    стабильные номера объектов, координаты и короткую траекторию.
    """

    def __init__(
        self,
        max_objects: int = MAX_TRACKED_OBJECTS,
        max_distance: float = TRACK_MAX_DISTANCE,
        max_lost_frames: int = TRACK_MAX_LOST_FRAMES,
        memory_frames: int = TRACK_MEMORY_FRAMES,
        memory_distance: float = TRACK_MEMORY_DISTANCE,
    ) -> None:
        self.max_objects = max_objects
        self.max_distance = max_distance
        self.max_lost_frames = max_lost_frames
        self.memory_frames = memory_frames
        self.memory_distance = memory_distance
        self.next_id = 1
        self.tracks: dict[int, Track] = {}
        self.track_memory: dict[int, TrackMemory] = {}

    def reset(self) -> None:
        self.next_id = 1
        self.tracks.clear()
        self.track_memory.clear()

    def update(self, detections: list[Detection]) -> list[Track]:
        """Обновляет треки новым набором детекций."""

        self._age_memory()

        if not detections:
            for track in self.tracks.values():
                track.lost_frames += 1
            self._drop_old_tracks()
            return []

        detections = sorted(detections, key=lambda item: item.confidence, reverse=True)

        if not self.tracks:
            for detection in detections[: self.max_objects]:
                self._register_or_restore(detection)
            return self.active_tracks

        # Связываем старые треки и новые детекции жадно, но аккуратно:
        # объект может продолжать только трек того же типа. Это сразу убирает
        # дикие перескоки вроде "телефон стал человеком".
        pairs: list[tuple[float, float, float, int, int]] = []
        for track_id, track in self.tracks.items():
            for detection_index, detection in enumerate(detections):
                if track.label != detection.label:
                    continue

                distance = hypot(
                    track.center[0] - detection.center[0],
                    track.center[1] - detection.center[1],
                )
                iou = bbox_iou(track.bbox, detection.bbox)
                score = distance - iou * 80.0
                pairs.append((score, distance, iou, track_id, detection_index))

        used_tracks: set[int] = set()
        used_detections: set[int] = set()

        for _, distance, iou, track_id, detection_index in sorted(pairs):
            if distance > self.max_distance and iou < TRACK_MIN_IOU:
                continue
            if track_id in used_tracks or detection_index in used_detections:
                continue
            self._refresh(track_id, detections[detection_index])
            used_tracks.add(track_id)
            used_detections.add(detection_index)

        for track_id, track in self.tracks.items():
            if track_id not in used_tracks:
                track.lost_frames += 1

        self._drop_old_tracks()

        for detection_index, detection in enumerate(detections):
            if detection_index in used_detections:
                continue
            if len(self.tracks) >= self.max_objects:
                break
            self._register_or_restore(detection)

        return self.active_tracks

    @property
    def active_tracks(self) -> list[Track]:
        tracks = [
            track
            for track in self.tracks.values()
            if track.lost_frames == 0 and track.confirmed
        ]
        return sorted(tracks, key=lambda item: item.track_id)[: self.max_objects]

    def _register(self, detection: Detection) -> None:
        track = Track(
            track_id=self.next_id,
            bbox=detection.bbox,
            label=detection.label,
            confidence=detection.confidence,
            center=detection.center,
        )
        track.trace.append(detection.center)
        self.tracks[self.next_id] = track
        self.next_id += 1

    def _register_or_restore(self, detection: Detection) -> None:
        if self._restore_from_memory(detection):
            return

        self._register(detection)

    def _restore_from_memory(self, detection: Detection) -> bool:
        memory_id = self._find_memory_match(detection)
        if memory_id is None:
            return False

        memory = self.track_memory.pop(memory_id)
        track = Track(
            track_id=memory.track_id,
            bbox=detection.bbox,
            label=detection.label,
            confidence=detection.confidence,
            center=detection.center,
            hits=max(memory.hits + 1, TRACK_MIN_HITS),
        )
        track.trace.extend(memory.trace)
        track.trace.append(detection.center)
        self.tracks[track.track_id] = track
        self.next_id = max(self.next_id, track.track_id + 1)
        return True

    def _find_memory_match(self, detection: Detection) -> int | None:
        best_match: tuple[float, int] | None = None

        for memory_id, memory in self.track_memory.items():
            if memory.label != detection.label:
                continue

            distance = hypot(
                memory.center[0] - detection.center[0],
                memory.center[1] - detection.center[1],
            )
            iou = bbox_iou(memory.bbox, detection.bbox)
            if distance > self.memory_distance and iou < TRACK_MIN_IOU:
                continue

            score = distance - iou * 80.0 + memory.age_frames * 0.25
            if best_match is None or score < best_match[0]:
                best_match = (score, memory_id)

        if best_match is None:
            return None
        return best_match[1]

    def _refresh(self, track_id: int, detection: Detection) -> None:
        track = self.tracks[track_id]
        track.bbox = detection.bbox
        track.label = detection.label
        track.confidence = detection.confidence
        track.center = detection.center
        track.hits += 1
        track.lost_frames = 0
        track.trace.append(detection.center)

    def _drop_old_tracks(self) -> None:
        for track_id in list(self.tracks):
            if self.tracks[track_id].lost_frames > self.max_lost_frames:
                self._remember_track(self.tracks[track_id])
                del self.tracks[track_id]

    def _remember_track(self, track: Track) -> None:
        self.track_memory[track.track_id] = TrackMemory(
            track_id=track.track_id,
            bbox=track.bbox,
            label=track.label,
            confidence=track.confidence,
            center=track.center,
            hits=track.hits,
            trace=tuple(track.trace),
        )

    def _age_memory(self) -> None:
        for memory_id in list(self.track_memory):
            memory = self.track_memory[memory_id]
            memory.age_frames += 1
            if memory.age_frames > self.memory_frames:
                del self.track_memory[memory_id]


def bbox_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0

    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union
