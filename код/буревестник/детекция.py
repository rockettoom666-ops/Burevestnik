from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from буревестник.настройки import (
    DETECTION_CONFIDENCE,
    DETECTION_IOU,
    DETECTION_MAX_RESULTS,
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
            Path.cwd() / configured_path,
            PROJECT_ROOT / configured_path,
            PROJECT_ROOT / "модели" / configured_path,
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
        ("Тест best_1.pt", PROJECT_ROOT / "модели" / "best_1.pt", 15),
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
    folders = [PROJECT_ROOT, PROJECT_ROOT / "модели", PROJECT_ROOT / "тест_модели"]

    for folder in folders:
        if not folder.exists():
            continue

        pattern = "*.pt" if folder == PROJECT_ROOT else "**/*.pt"
        for path in folder.glob(pattern):
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

    result = model.predict(
        source=frame,
        conf=DETECTION_CONFIDENCE,
        iou=DETECTION_IOU,
        max_det=DETECTION_MAX_RESULTS,
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
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0]]
        detections.append(
            Detection(
                bbox=(x1, y1, x2, y2),
                label=raw_label,
                confidence=confidence,
            )
        )

    return detections
