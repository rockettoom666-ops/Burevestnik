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

# Настройки поиска дальних темных точек. Это не "готовая детекция дрона",
# а аккуратный сигнал оператору: "здесь есть маленький темный объект,
# который YOLO может не распознать".
SUSPICIOUS_DOT_DARK_THRESHOLD = 70
SUSPICIOUS_DOT_MIN_AREA = 10
SUSPICIOUS_DOT_MAX_AREA = 120
SUSPICIOUS_DOT_MIN_WIDTH = 4
SUSPICIOUS_DOT_MIN_HEIGHT = 4
SUSPICIOUS_DOT_MAX_WIDTH = 26
SUSPICIOUS_DOT_MAX_HEIGHT = 26
SUSPICIOUS_DOT_IGNORE_RADIUS = 48
SUSPICIOUS_DOT_CONFIRM_RADIUS = 70
SUSPICIOUS_DOT_DEFAULT_LABEL = "drone?"
SUSPICIOUS_DOT_MATCH_RADIUS = 36
SUSPICIOUS_DOT_MIN_FRAMES = 4
SUSPICIOUS_DOT_MIN_MOVE = 10.0
SUSPICIOUS_DOT_MIN_PATH = 14.0
SUSPICIOUS_DOT_MAX_MISSED_FRAMES = 3
SUSPICIOUS_DOT_TRAJECTORY_LENGTH = 8

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
