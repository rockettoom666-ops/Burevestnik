from __future__ import annotations

from typing import Any

from буревестник.настройки import (
    DETECTION_CONFIDENCE,
    YOLO_LABELS_RU,
    YOLO_MODEL_NAME,
)
from буревестник.сущности import Detection


def load_yolo_model() -> Any:
    """Загружает YOLO только когда пользователь включает отслеживание."""

    from ultralytics import YOLO

    return YOLO(YOLO_MODEL_NAME)


def detect_objects(frame, model: Any) -> list[Detection]:
    """Запускает YOLO и превращает результат в простой список Detection."""

    result = model.predict(
        source=frame,
        conf=DETECTION_CONFIDENCE,
        verbose=False,
    )[0]

    detections: list[Detection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    names = getattr(model, "names", {}) or {}

    for box in boxes:
        confidence = float(box.conf[0])
        if confidence < DETECTION_CONFIDENCE:
            continue

        class_id = int(box.cls[0])
        raw_label = str(names.get(class_id, f"class_{class_id}"))
        label = translate_label(raw_label)
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0]]
        detections.append(
            Detection(
                bbox=(x1, y1, x2, y2),
                label=label,
                confidence=confidence,
            )
        )

    return detections


def translate_label(label: str) -> str:
    return YOLO_LABELS_RU.get(label, label)

