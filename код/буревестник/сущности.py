from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


# Одна найденная камера: ее номер в системе и размер первого полученного кадра.
# Название устройства OpenCV обычно не дает, поэтому в интерфейсе показываем
# понятное "Камера 0", "Камера 1" и разрешение.
@dataclass(frozen=True)
class CameraInfo:
    index: int
    width: int
    height: int

    @property
    def title(self) -> str:
        return f"Камера {self.index}"

    @property
    def details(self) -> str:
        return f"{self.width} x {self.height}"


@dataclass(frozen=True)
class Detection:
    """Одна находка YOLO на текущем кадре."""

    bbox: tuple[int, int, int, int]
    label: str
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2


@dataclass
class Track:
    """Объект, который мы пытаемся удерживать между соседними кадрами."""

    track_id: int
    bbox: tuple[int, int, int, int]
    label: str
    confidence: float
    center: tuple[int, int]
    hits: int = 1
    lost_frames: int = 0
    trace: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=30))

    @property
    def confirmed(self) -> bool:
        from буревестник.настройки import TRACK_MIN_HITS

        return self.hits >= TRACK_MIN_HITS
