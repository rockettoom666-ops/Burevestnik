from __future__ import annotations

from math import hypot

from буревестник.настройки import (
    MAX_TRACKED_OBJECTS,
    TRACK_MAX_DISTANCE,
    TRACK_MAX_LOST_FRAMES,
)
from буревестник.сущности import Detection, Track


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
    ) -> None:
        self.max_objects = max_objects
        self.max_distance = max_distance
        self.max_lost_frames = max_lost_frames
        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    def reset(self) -> None:
        self.next_id = 1
        self.tracks.clear()

    def update(self, detections: list[Detection]) -> list[Track]:
        """Обновляет треки новым набором детекций."""

        if not detections:
            for track in self.tracks.values():
                track.lost_frames += 1
            self._drop_old_tracks()
            return []

        detections = sorted(detections, key=lambda item: item.confidence, reverse=True)

        if not self.tracks:
            for detection in detections[: self.max_objects]:
                self._register(detection)
            return self.active_tracks

        # Связываем старые треки и новые детекции жадно по расстоянию центров.
        pairs: list[tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            for detection_index, detection in enumerate(detections):
                distance = hypot(
                    track.center[0] - detection.center[0],
                    track.center[1] - detection.center[1],
                )
                pairs.append((distance, track_id, detection_index))

        used_tracks: set[int] = set()
        used_detections: set[int] = set()

        for distance, track_id, detection_index in sorted(pairs):
            if distance > self.max_distance:
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
            self._register(detection)

        return self.active_tracks

    @property
    def active_tracks(self) -> list[Track]:
        tracks = [track for track in self.tracks.values() if track.lost_frames == 0]
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

    def _refresh(self, track_id: int, detection: Detection) -> None:
        track = self.tracks[track_id]
        track.bbox = detection.bbox
        track.label = detection.label
        track.confidence = detection.confidence
        track.center = detection.center
        track.lost_frames = 0
        track.trace.append(detection.center)

    def _drop_old_tracks(self) -> None:
        for track_id in list(self.tracks):
            if self.tracks[track_id].lost_frames > self.max_lost_frames:
                del self.tracks[track_id]

