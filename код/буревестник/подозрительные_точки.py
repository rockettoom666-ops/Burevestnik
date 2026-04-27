from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot

import cv2

from буревестник.настройки import (
    SUSPICIOUS_DOT_DARK_THRESHOLD,
    SUSPICIOUS_DOT_MAX_AREA,
    SUSPICIOUS_DOT_MAX_HEIGHT,
    SUSPICIOUS_DOT_MAX_MISSED_FRAMES,
    SUSPICIOUS_DOT_MAX_WIDTH,
    SUSPICIOUS_DOT_MATCH_RADIUS,
    SUSPICIOUS_DOT_MIN_AREA,
    SUSPICIOUS_DOT_MIN_FRAMES,
    SUSPICIOUS_DOT_MIN_HEIGHT,
    SUSPICIOUS_DOT_MIN_MOVE,
    SUSPICIOUS_DOT_MIN_PATH,
    SUSPICIOUS_DOT_MIN_WIDTH,
    SUSPICIOUS_DOT_TRAJECTORY_LENGTH,
)
from буревестник.сущности import Detection


@dataclass(frozen=True)
class SuspiciousPoint:
    """Маленькая темная точка, которую стоит показать оператору."""

    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    area: float
    darkness: int
    trajectory: tuple[tuple[int, int], ...] = ()

    @property
    def score(self) -> float:
        # Чем темнее и компактнее точка, тем выше она в списке кандидатов.
        return self.darkness + min(self.area, 40.0)


@dataclass
class ConfirmedSuspiciousTarget:
    """Точка, которую оператор подтвердил и подписал вручную."""

    label: str
    center: tuple[int, int]
    missed_frames: int = 0


@dataclass
class SuspiciousMotionCandidate:
    """Кандидат на дальнюю цель, который проверяется по движению."""

    candidate_id: int
    point: SuspiciousPoint
    centers: deque[tuple[int, int]] = field(
        default_factory=lambda: deque(maxlen=SUSPICIOUS_DOT_TRAJECTORY_LENGTH)
    )
    hits: int = 1
    missed_frames: int = 0

    def __post_init__(self) -> None:
        self.centers.append(self.point.center)

    def refresh(self, point: SuspiciousPoint) -> None:
        self.point = point
        self.centers.append(point.center)
        self.hits += 1
        self.missed_frames = 0

    @property
    def movement(self) -> float:
        if len(self.centers) < 2:
            return 0.0

        first = self.centers[0]
        last = self.centers[-1]
        return hypot(first[0] - last[0], first[1] - last[1])

    @property
    def path_length(self) -> float:
        if len(self.centers) < 2:
            return 0.0

        points = list(self.centers)
        return sum(
            hypot(points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1])
            for index in range(1, len(points))
        )

    @property
    def ready_for_operator(self) -> bool:
        return (
            self.hits >= SUSPICIOUS_DOT_MIN_FRAMES
            and self.movement >= SUSPICIOUS_DOT_MIN_MOVE
            and self.path_length >= SUSPICIOUS_DOT_MIN_PATH
        )

    def as_point(self) -> SuspiciousPoint:
        return SuspiciousPoint(
            bbox=self.point.bbox,
            center=self.point.center,
            area=self.point.area,
            darkness=self.point.darkness,
            trajectory=tuple(self.centers),
        )


class SuspiciousMotionTracker:
    """Отсекает одиночные точки и оставляет только движущиеся кандидаты."""

    def __init__(self) -> None:
        self.next_id = 1
        self.candidates: dict[int, SuspiciousMotionCandidate] = {}

    def reset(self) -> None:
        self.next_id = 1
        self.candidates.clear()

    def update(self, points: list[SuspiciousPoint]) -> list[SuspiciousPoint]:
        used_candidates: set[int] = set()

        for point in sorted(points, key=lambda item: item.score, reverse=True):
            candidate_id = self._nearest_candidate(point.center, used_candidates)
            if candidate_id is None:
                self._register(point, used_candidates)
            else:
                self.candidates[candidate_id].refresh(point)
                used_candidates.add(candidate_id)

        for candidate_id in list(self.candidates):
            if candidate_id in used_candidates:
                continue

            candidate = self.candidates[candidate_id]
            candidate.missed_frames += 1
            if candidate.missed_frames > SUSPICIOUS_DOT_MAX_MISSED_FRAMES:
                del self.candidates[candidate_id]

        ready_points = [
            candidate.as_point()
            for candidate in self.candidates.values()
            if candidate.ready_for_operator
        ]
        return sorted(ready_points, key=lambda item: item.score, reverse=True)

    def _nearest_candidate(
        self,
        center: tuple[int, int],
        used_candidates: set[int],
    ) -> int | None:
        best_id: int | None = None
        best_distance = SUSPICIOUS_DOT_MATCH_RADIUS

        for candidate_id, candidate in self.candidates.items():
            if candidate_id in used_candidates:
                continue

            distance = hypot(
                center[0] - candidate.point.center[0],
                center[1] - candidate.point.center[1],
            )
            if distance <= best_distance:
                best_id = candidate_id
                best_distance = distance

        return best_id

    def _register(
        self,
        point: SuspiciousPoint,
        used_candidates: set[int],
    ) -> None:
        self.candidates[self.next_id] = SuspiciousMotionCandidate(
            candidate_id=self.next_id,
            point=point,
        )
        used_candidates.add(self.next_id)
        self.next_id += 1


def find_suspicious_points(
    frame,
    known_detections: list[Detection],
) -> list[SuspiciousPoint]:
    """Ищет маленькие темные точки, которые YOLO мог пропустить.

    Мы специально пропускаем точки внутри уже известных рамок YOLO, чтобы не
    тревожить оператора из-за глаз, кнопок, камеры телефона и других деталей.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, mask = cv2.threshold(
        gray,
        SUSPICIOUS_DOT_DARK_THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points: list[SuspiciousPoint] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < SUSPICIOUS_DOT_MIN_AREA or area > SUSPICIOUS_DOT_MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < SUSPICIOUS_DOT_MIN_WIDTH or h < SUSPICIOUS_DOT_MIN_HEIGHT:
            continue

        if w > SUSPICIOUS_DOT_MAX_WIDTH or h > SUSPICIOUS_DOT_MAX_HEIGHT:
            continue

        aspect = w / max(h, 1)
        if aspect < 0.35 or aspect > 2.8:
            continue

        cx, cy = x + w // 2, y + h // 2
        if _inside_any_detection((cx, cy), known_detections):
            continue

        patch = gray[y : y + h, x : x + w]
        darkness = 255 - int(patch.mean()) if patch.size else 0
        points.append(
            SuspiciousPoint(
                bbox=(x, y, x + w, y + h),
                center=(cx, cy),
                area=area,
                darkness=darkness,
            )
        )

    return sorted(points, key=lambda item: item.score, reverse=True)


def nearest_point(
    center: tuple[int, int],
    points: list[SuspiciousPoint],
    radius: float,
) -> SuspiciousPoint | None:
    best: SuspiciousPoint | None = None
    best_distance = radius

    for point in points:
        distance = hypot(center[0] - point.center[0], center[1] - point.center[1])
        if distance <= best_distance:
            best = point
            best_distance = distance

    return best


def point_near_any(
    center: tuple[int, int],
    zones: list[tuple[int, int]],
    radius: float,
) -> bool:
    return any(hypot(center[0] - x, center[1] - y) <= radius for x, y in zones)


def make_detection(point: SuspiciousPoint, label: str) -> Detection:
    x1, y1, x2, y2 = point.bbox
    padding = 8
    return Detection(
        bbox=(x1 - padding, y1 - padding, x2 + padding, y2 + padding),
        label=label,
        confidence=0.99,
    )


def _inside_any_detection(
    center: tuple[int, int],
    detections: list[Detection],
) -> bool:
    cx, cy = center

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        margin = 8
        if x1 - margin <= cx <= x2 + margin and y1 - margin <= cy <= y2 + margin:
            return True

    return False
