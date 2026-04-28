from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from буревестник.настройки import (
    CLASS_CONFIDENCE_THRESHOLDS,
    DETECTION_CONFIDENCE,
    DETECTION_IOU,
    DETECTION_LOW_CONFIDENCE,
    DETECTION_MAX_RESULTS,
    SMALL_OBJECT_AREA,
    SMALL_OBJECT_CONFIDENCE_RELAXATION,
    TARGET_CLASSES,
    VIDEO_ANALYSIS_FRAME_INTERVAL,
    YOLO_MODEL_ENV_VAR,
    YOLO_MODEL_NAME,
)
from буревестник.сущности import Detection


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class YoloModelChoice:
    """Одна версия модели, которую можно выбрать в отладке."""

    label: str
    path: Path
    video_interval: int


def resolve_yolo_model_path(model_name: str | Path | None = None) -> Path:
    """Находит файл обученной модели независимо от папки запуска."""

    configured_name = str(model_name or os.environ.get(YOLO_MODEL_ENV_VAR, YOLO_MODEL_NAME))
    configured_path = Path(configured_name)
    if configured_path.is_absolute():
        candidates = [configured_path]
    else:
        candidates = [
            PROJECT_ROOT / "модели" / configured_path,
            PROJECT_ROOT / configured_path,
            Path.cwd() / configured_path,
            Path(__file__).resolve().parents[1] / configured_path,
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked_places = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        f"Не найден файл обученной модели {configured_name}.\n"
        f"Положи модель в корень проекта, в папку \"модели\" или укажи путь через {YOLO_MODEL_ENV_VAR}.\n\n"
        f"Проверенные места:\n{checked_places}"
    )


def list_yolo_model_choices() -> list[YoloModelChoice]:
    """Собирает версии моделей, которые можно выбрать прямо в интерфейсе."""

    raw_choices = [
        ("Основная best.pt", YOLO_MODEL_NAME, VIDEO_ANALYSIS_FRAME_INTERVAL),
        ("Модель best_1.pt", PROJECT_ROOT / "модели" / "best_1.pt", 15),
    ]

    env_model = os.environ.get(YOLO_MODEL_ENV_VAR)
    if env_model:
        raw_choices.insert(0, (f"Из окружения {Path(env_model).name}", env_model, VIDEO_ANALYSIS_FRAME_INTERVAL))

    found_paths = set()
    choices: list[YoloModelChoice] = []

    for label, model_name, interval in raw_choices:
        try:
            path = resolve_yolo_model_path(model_name).resolve()
        except FileNotFoundError:
            continue

        if path in found_paths:
            continue

        found_paths.add(path)
        choices.append(YoloModelChoice(label=label, path=path, video_interval=interval))

    for path in _scan_extra_model_files(found_paths):
        choices.append(
            YoloModelChoice(
                label=f"Найдена {path.name}",
                path=path,
                video_interval=VIDEO_ANALYSIS_FRAME_INTERVAL,
            )
        )

    return choices


def _scan_extra_model_files(known_paths: set[Path]) -> list[Path]:
    model_files: list[Path] = []
    folders = [PROJECT_ROOT / "модели"]

    for folder in folders:
        if not folder.exists():
            continue

        for path in folder.glob("**/*.pt"):
            resolved = path.resolve()
            if resolved not in known_paths:
                model_files.append(resolved)
                known_paths.add(resolved)

    return sorted(model_files, key=lambda item: item.name.lower())


def load_yolo_model(model_name: str | Path | None = None) -> Any:
    """Загружает YOLO только когда пользователь включает отслеживание."""

    from ultralytics import YOLO

    model_path = resolve_yolo_model_path(model_name)
    return YOLO(str(model_path))


def get_yolo_model_title(model_name: str | Path | None = None) -> str:
    """Возвращает короткое имя модели для сообщений в интерфейсе."""

    return resolve_yolo_model_path(model_name).name


def detect_objects(frame, model: Any) -> list[Detection]:
    """Запускает YOLO и превращает результат в простой список Detection."""

    # Напарник усиливал кадр перед моделью. Оставляем это здесь, чтобы GUI не знал
    # про техническую кухню детекции и просто передавал обычный кадр.
    enhanced_frame = _prepare_frame_for_yolo(frame)

    detections = _run_yolo_pass(
        enhanced_frame,
        model,
        min_confidence=DETECTION_CONFIDENCE,
        augment=False,
        use_class_thresholds=True,
    )
    if not detections:
        detections = _run_yolo_pass(
            enhanced_frame,
            model,
            min_confidence=DETECTION_LOW_CONFIDENCE,
            # В файле напарника здесь был augment=True, но на видеопотоке это сильно
            # тормозит. Для прототипа лучше оставить живую картинку и мягче снизить порог.
            augment=False,
            use_class_thresholds=False,
        )

    return _remove_duplicate_detections(detections)


def _prepare_frame_for_yolo(frame):
    """Чуть вытягивает контраст и резкость перед запуском обученной модели."""

    sharpen_kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )
    sharp = cv2.filter2D(frame, -1, sharpen_kernel)
    lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    light, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    light = clahe.apply(light)
    return cv2.cvtColor(cv2.merge((light, channel_a, channel_b)), cv2.COLOR_LAB2BGR)


def _run_yolo_pass(
    frame,
    model: Any,
    min_confidence: float,
    augment: bool,
    use_class_thresholds: bool,
) -> list[Detection]:
    """Один прогон YOLO с фильтрацией классов и уверенности."""

    result = model.predict(
        source=frame,
        conf=min_confidence,
        iou=DETECTION_IOU,
        max_det=DETECTION_MAX_RESULTS,
        augment=augment,
        verbose=False,
    )[0]

    detections: list[Detection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    names = getattr(model, "names", {}) or {}

    for box in boxes:
        score = float(box.conf[0])
        if score < min_confidence:
            continue

        class_id = int(box.cls[0])
        raw_label = str(names.get(class_id, f"class_{class_id}"))
        if TARGET_CLASSES and raw_label not in TARGET_CLASSES:
            continue

        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0]]
        if use_class_thresholds:
            class_confidence = CLASS_CONFIDENCE_THRESHOLDS.get(raw_label, min_confidence)
            area = max(x2 - x1, 0) * max(y2 - y1, 0)
            if area < SMALL_OBJECT_AREA:
                class_confidence -= SMALL_OBJECT_CONFIDENCE_RELAXATION

            if score < class_confidence:
                continue

        detections.append(
            Detection(
                bbox=(x1, y1, x2, y2),
                label=raw_label,
                confidence=score,
            )
        )

    return detections


def _remove_duplicate_detections(detections: list[Detection]) -> list[Detection]:
    """Убирает почти одинаковые рамки одного класса, оставляя более уверенную."""

    if len(detections) < 2:
        return detections

    indexes_to_remove: set[int] = set()
    for left_index, left in enumerate(detections):
        if left_index in indexes_to_remove:
            continue

        for right_index in range(left_index + 1, len(detections)):
            if right_index in indexes_to_remove:
                continue

            right = detections[right_index]
            if left.label != right.label:
                continue

            if _bbox_iou(left.bbox, right.bbox) <= 0.5:
                continue

            if left.confidence < right.confidence:
                indexes_to_remove.add(left_index)
                break

            indexes_to_remove.add(right_index)

    return [
        detection
        for index, detection in enumerate(detections)
        if index not in indexes_to_remove
    ]


def _bbox_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1)
