from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import cv2

from буревестник.настройки import (
    SUSPICIOUS_DOT_DARK_THRESHOLD,
    SUSPICIOUS_DOT_MAX_AREA,
    SUSPICIOUS_DOT_MAX_HEIGHT,
    SUSPICIOUS_DOT_MAX_WIDTH,
    SUSPICIOUS_DOT_MIN_AREA,
)
from буревестник.сущности import Detection


@dataclass(frozen=True)
class SuspiciousPoint:
    """Маленькая темная точка, которую стоит показать оператору."""

    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    area: float
    darkness: int

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
