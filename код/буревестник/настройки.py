from __future__ import annotations

# Настройки первого рабочего трекера.
# Сейчас специально берем все классы YOLO, чтобы прототип можно было проверить
# даже на обычной веб-камере: человек, бутылка или стул тоже получат ID.
# Для финала сузим список до воздушных объектов и обученной модели дронов.
MAX_TRACKED_OBJECTS = 5
DETECTION_CONFIDENCE = 0.50
DETECTION_IOU = 0.45
DETECTION_MAX_RESULTS = 8
TRACK_MAX_DISTANCE = 70.0
TRACK_MAX_LOST_FRAMES = 8
TRACK_MIN_HITS = 2
TRACK_MIN_IOU = 0.08

YOLO_MODEL_NAME = "yolov8n.pt"

YOLO_LABELS_RU = {
    "person": "человек",
    "bicycle": "велосипед",
    "car": "машина",
    "motorcycle": "мотоцикл",
    "airplane": "самолет",
    "bus": "автобус",
    "train": "поезд",
    "truck": "грузовик",
    "boat": "лодка",
    "bird": "птица",
    "cat": "кот",
    "dog": "собака",
    "backpack": "рюкзак",
    "umbrella": "зонт",
    "bottle": "бутылка",
    "chair": "стул",
    "laptop": "ноутбук",
    "cell phone": "телефон",
}

TRACK_COLORS = [
    (0, 220, 255),
    (80, 220, 80),
    (255, 160, 60),
    (220, 100, 255),
    (255, 220, 80),
]
