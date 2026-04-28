from __future__ import annotations

# Настройки первого рабочего трекера.
# Теперь основной детектор - обученная модель напарника. Файл best.pt должен
# лежать в корне проекта или в папке "модели".
MAX_TRACKED_OBJECTS = 5
DETECTION_CONFIDENCE = 0.40
DETECTION_LOW_CONFIDENCE = 0.30
DETECTION_IOU = 0.45
DETECTION_MAX_RESULTS = 8
SMALL_OBJECT_AREA = 5000
SMALL_OBJECT_CONFIDENCE_RELAXATION = 0.10
VIDEO_ANALYSIS_FRAME_INTERVAL = 5
TRACK_MAX_DISTANCE = 70.0
TRACK_MAX_LOST_FRAMES = 8
TRACK_MEMORY_FRAMES = 45
TRACK_MEMORY_DISTANCE = 55.0
TRACK_MIN_HITS = 2
TRACK_MIN_IOU = 0.08

YOLO_MODEL_NAME = "best.pt"
YOLO_MODEL_ENV_VAR = "BUREVESTNIK_MODEL"

TARGET_CLASSES = {
    "aerial-object",
    "drone",
    "Drone",
    "uav",
    "UAV",
    "БПЛА",
    "airplane",
    "helicopter",
}

CLASS_CONFIDENCE_THRESHOLDS = {
    "aerial-object": 0.40,
    "drone": 0.45,
    "Drone": 0.45,
    "uav": 0.45,
    "UAV": 0.45,
    "БПЛА": 0.45,
    "airplane": 0.40,
    "helicopter": 0.35,
}

YOLO_LABELS_RU = {
    "aerial-object": "воздушная цель",
    "drone": "дрон",
    "Drone": "дрон",
    "uav": "дрон",
    "UAV": "дрон",
    "БПЛА": "дрон",
    "person": "человек",
    "bicycle": "велосипед",
    "car": "машина",
    "motorcycle": "мотоцикл",
    "airplane": "самолет",
    "helicopter": "вертолет",
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
