from __future__ import annotations

import cv2

from буревестник.настройки import MAX_TRACKED_OBJECTS, TRACK_COLORS
from буревестник.подозрительные_точки import SuspiciousPoint
from буревестник.сущности import Track


def draw_tracking_overlay(frame, tracks: list[Track]) -> None:
    """Рисует рамки, ID, центр и траекторию поверх кадра."""

    for track in tracks[:MAX_TRACKED_OBJECTS]:
        x1, y1, x2, y2 = track.bbox
        cx, cy = track.center

        color = color_for_id(track.track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 4, color, -1)

        title = f"ID {track.track_id} | {safe_cv_text(track.label)} | {cx},{cy}"
        cv2.putText(
            frame,
            title,
            (x1, max(22, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )

        confidence = f"{track.confidence * 100:.0f}%"
        cv2.putText(
            frame,
            confidence,
            (x1, min(frame.shape[0] - 8, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        if len(track.trace) > 1:
            points = list(track.trace)
            for index in range(1, len(points)):
                cv2.line(frame, points[index - 1], points[index], color, 2, cv2.LINE_AA)


def draw_waiting_label(frame, text: str) -> None:
    """Пишет поверх кадра служебный текст, пока модель не готова."""

    cv2.rectangle(frame, (16, 16), (390, 58), (15, 23, 42), -1)
    cv2.putText(
        frame,
        text,
        (28, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_suspicious_point(frame, point: SuspiciousPoint) -> None:
    """Подсвечивает точку, по которой оператор должен принять решение."""

    x1, y1, x2, y2 = point.bbox
    cx, cy = point.center
    color = (0, 255, 255)

    cv2.rectangle(frame, (x1 - 8, y1 - 8), (x2 + 8, y2 + 8), color, 2)
    cv2.circle(frame, (cx, cy), 12, color, 2)

    if len(point.trajectory) > 1:
        points = list(point.trajectory)
        for index in range(1, len(points)):
            cv2.line(frame, points[index - 1], points[index], color, 2, cv2.LINE_AA)

    cv2.putText(
        frame,
        "possible drone?",
        (max(8, x1 - 8), max(24, y1 - 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def color_for_id(track_id: int) -> tuple[int, int, int]:
    return TRACK_COLORS[(track_id - 1) % len(TRACK_COLORS)]


def safe_cv_text(text: str) -> str:
    """OpenCV не рисует кириллицу, поэтому на кадре оставляем безопасный текст."""

    ascii_text = text.encode("ascii", errors="ignore").decode("ascii").strip()
    return ascii_text or "target"
