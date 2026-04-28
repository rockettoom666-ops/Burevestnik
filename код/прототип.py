from __future__ import annotations

"""
Система распознавания и сопровождения воздушных объектов "Буревестник".
Финальная версия:
- Раздельные пороги уверенности
- Коррекция конфликтов drone vs helicopter
- Динамическая ROI, пауза, временная фильтрация уверенности
- Удаление дубликатов только одинаковых классов
- Синхронизация таблицы и отрисовки
"""

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# =========================
# КОНФИГУРАЦИЯ
# =========================
MAX_OBJECTS = 5
CONF_THRESHOLD = 0.4
CONF_LOW = 0.3
TARGET_CLASSES = ["drone", "airplane", "helicopter"]
MAX_TRACK_AGE = 60
TRACK_DIST_THRESH = 80

CLASS_THRESHOLDS = {
    "drone": 0.45,
    "airplane": 0.4,
    "helicopter": 0.35,
}

# =========================
# КЛАСС ТРЕКА
# =========================
class Track:
    def __init__(self, track_id, bbox, label="unknown", conf=0.0):
        self.id = track_id
        self.bbox = bbox
        self.label = label
        self.conf = conf
        self.conf_history = deque(maxlen=5)
        self.conf_history.append(conf)
        self.centers = deque(maxlen=30)
        self.age = 1
        self.unseen = 0
        self.hits = 1
        self.confirmed = False

        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.P *= 10.0
        self.kf.R = np.diag([5.0, 5.0])
        self.kf.Q = np.diag([1.0, 1.0, 1.0, 1.0])

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        self.kf.x = np.array([cx, cy, 0.0, 0.0])
        self.centroid = (int(cx), int(cy))
        self.centers.append(self.centroid)

    def predict(self):
        self.kf.predict()
        pred_x, pred_y = self.kf.x[0], self.kf.x[1]
        return int(pred_x), int(pred_y)

    def update(self, bbox, label, conf=0.0):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        self.kf.update(np.array([cx, cy]))
        self.bbox = bbox
        self.label = label
        self.conf = conf
        self.conf_history.append(conf)
        self.centroid = (int(self.kf.x[0]), int(self.kf.x[1]))
        self.centers.append(self.centroid)
        self.unseen = 0
        self.age += 1
        self.hits += 1
        if self.hits >= 3:
            self.confirmed = True

    def mark_missed(self):
        self.unseen += 1
        self.kf.predict()
        self.centroid = (int(self.kf.x[0]), int(self.kf.x[1]))

    def is_dead(self):
        return self.unseen > MAX_TRACK_AGE

    @property
    def velocity(self):
        vx, vy = self.kf.x[2], self.kf.x[3]
        return vx, vy

    @property
    def avg_conf(self):
        if not self.conf_history:
            return 0.0
        return sum(self.conf_history) / len(self.conf_history)


# =========================
# ТРЕКЕР
# =========================
class KalmanTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def update(self, detections, labels, confidences=None):
        for tr in self.tracks:
            tr.predict()

        cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for i, tr in enumerate(self.tracks):
            pred_centroid = (int(tr.kf.x[0]), int(tr.kf.x[1]))
            for j, det_bbox in enumerate(detections):
                x1, y1, x2, y2 = det_bbox
                det_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                dist = np.linalg.norm(np.array(pred_centroid) - np.array(det_center))
                if confidences and confidences[j] > 0:
                    dist = dist / (confidences[j] + 0.1)
                cost_matrix[i, j] = dist

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_tracks = set()
        assigned_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < TRACK_DIST_THRESH:
                self.tracks[r].update(detections[c], labels[c],
                                      confidences[c] if confidences else 0.0)
                assigned_tracks.add(r)
                assigned_dets.add(c)

        for i, tr in enumerate(self.tracks):
            if i not in assigned_tracks:
                tr.mark_missed()

        for j, det in enumerate(detections):
            if j not in assigned_dets:
                new_track = Track(self.next_id, det, labels[j],
                                  confidences[j] if confidences else 0.0)
                self.tracks.append(new_track)
                self.next_id += 1

        self.tracks = [tr for tr in self.tracks if not tr.is_dead()]
        return {tr.id: tr for tr in self.tracks}


# =========================
# ОЦЕНКА УГРОЗЫ
# =========================
def compute_threat(track, frame_center, roi=None):
    cx, cy = track.centroid
    dist = np.hypot(cx - frame_center[0], cy - frame_center[1])
    vx, vy = track.velocity
    speed = np.hypot(vx, vy)
    threat = (1000 - dist) * 0.5 + speed * 10
    if roi is not None:
        rx1, ry1, rx2, ry2 = roi
        if rx1 < cx < rx2 and ry1 < cy < ry2:
            threat += 500
    if track.label in ("airplane", "helicopter"):
        threat += 200
    elif track.label == "drone":
        threat += 400
    elif track.label == "bird":
        threat -= 150
    return threat


# =========================
# ИНФОРМАЦИЯ О КАМЕРЕ
# =========================
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


# =========================
# ГЛАВНОЕ ПРИЛОЖЕНИЕ
# =========================
class BurevestnikSystem(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Буревестник – система обнаружения и сопровождения")
        self.geometry("1280x800")
        self.minsize(1060, 680)
        self.configure(fg_color="#0b1018")

        self.capture: cv2.VideoCapture | None = None
        self.frame_job: str | None = None
        self.current_photo: ImageTk.PhotoImage | None = None
        self.current_source = "Источник не выбран"
        self.current_mode = "Ожидание"
        self.current_resolution = "-"

        self.video_delay_ms = 15
        self.last_frame_at = time.perf_counter()
        self.smoothed_fps = 0.0
        self.scanning = False
        self.paused = False

        self.model: YOLO | None = None
        self.tracker = KalmanTracker()
        self.roi = None
        self.roi_start = None
        self.selecting_roi = False
        self.show_detections = True
        self.last_detection_count = 0

        self.last_gray = None
        self.roi_anchors = None
        self.roi_anchor_frame = None
        self.last_raw_frame = None

        self.bind("<KeyPress-z>", lambda e: self.enable_roi_selection())
        self.bind("<KeyPress-r>", lambda e: self.reset_roi())

        self.after(100, self._load_model)

        self.table_text: ctk.CTkTextbox | None = None
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        self._build_layout()
        self.show_empty_view()

    def _load_model(self):
        try:
            self.status_label.configure(text="Загружаю модель...")
            self.model = YOLO("best.pt")
            self.status_label.configure(text="Модель загружена. Готов к выбору источника.")
        except Exception as e:
            self.status_label.configure(text=f"Ошибка загрузки модели: {e}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель YOLO:\n{e}")

    # ========== ИНТЕРФЕЙС ==========
    def _build_layout(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=350, corner_radius=0, fg_color="#111827")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(9, weight=1)

        self.main = ctk.CTkFrame(self, corner_radius=0, fg_color="#0b1018")
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(1, weight=1)

        title = ctk.CTkLabel(self.sidebar, text="Буревестник", font=ctk.CTkFont(size=28, weight="bold"), text_color="#f8fafc")
        title.grid(row=0, column=0, padx=24, pady=(28, 0), sticky="w")

        subtitle = ctk.CTkLabel(self.sidebar, text="Обнаружение и сопровождение\nвоздушных объектов", justify="left", font=ctk.CTkFont(size=14), text_color="#9ca3af")
        subtitle.grid(row=1, column=0, padx=24, pady=(4, 24), sticky="w")

        self.camera_button = ctk.CTkButton(self.sidebar, text="Работать с камерой", height=46, corner_radius=10, fg_color="#0ea5e9", hover_color="#0284c7", font=ctk.CTkFont(size=15, weight="bold"), command=self.scan_cameras)
        self.camera_button.grid(row=2, column=0, padx=24, pady=(0, 12), sticky="ew")

        self.video_button = ctk.CTkButton(self.sidebar, text="Тест на видео", height=46, corner_radius=10, fg_color="#14b8a6", hover_color="#0f766e", font=ctk.CTkFont(size=15, weight="bold"), command=self.open_video_dialog)
        self.video_button.grid(row=3, column=0, padx=24, pady=(0, 12), sticky="ew")

        self.pause_button = ctk.CTkButton(self.sidebar, text="Пауза", height=40, corner_radius=10, fg_color="#f59e0b", hover_color="#d97706", command=self.toggle_pause)
        self.pause_button.grid(row=4, column=0, padx=24, pady=(0, 6), sticky="ew")

        self.stop_button = ctk.CTkButton(self.sidebar, text="Остановить", height=36, corner_radius=10, fg_color="#374151", hover_color="#4b5563", command=self.stop_source)
        self.stop_button.grid(row=5, column=0, padx=24, pady=(0, 12), sticky="ew")

        self.detect_switch = ctk.CTkSwitch(self.sidebar, text="Показывать объекты", command=self.toggle_detections, onvalue=True, offvalue=False)
        self.detect_switch.select()
        self.detect_switch.grid(row=6, column=0, padx=24, pady=(0, 12), sticky="w")

        roi_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        roi_frame.grid(row=7, column=0, padx=24, pady=(0, 12), sticky="ew")
        roi_frame.grid_columnconfigure(0, weight=1)
        roi_frame.grid_columnconfigure(1, weight=1)

        self.roi_button = ctk.CTkButton(roi_frame, text="Задать зону (Z)", height=36, corner_radius=8, fg_color="#3b82f6", hover_color="#2563eb", command=self.enable_roi_selection)
        self.roi_button.grid(row=0, column=0, padx=(0, 6), sticky="ew")
        self.roi_reset_button = ctk.CTkButton(roi_frame, text="Сбросить (R)", height=36, corner_radius=8, fg_color="#4b5563", hover_color="#6b7280", command=self.reset_roi)
        self.roi_reset_button.grid(row=0, column=1, padx=(6, 0), sticky="ew")

        cameras_title = ctk.CTkLabel(self.sidebar, text="Найденные камеры", font=ctk.CTkFont(size=16, weight="bold"), text_color="#e5e7eb")
        cameras_title.grid(row=8, column=0, padx=24, pady=(8, 8), sticky="w")

        self.camera_list = ctk.CTkScrollableFrame(self.sidebar, fg_color="#0f172a", corner_radius=12, height=120)
        self.camera_list.grid(row=9, column=0, padx=24, pady=(0, 12), sticky="ew")
        self.camera_list.grid_columnconfigure(0, weight=1)

        table_title = ctk.CTkLabel(self.sidebar, text="Сопровождение (таблица)", font=ctk.CTkFont(size=16, weight="bold"), text_color="#e5e7eb")
        table_title.grid(row=10, column=0, padx=24, pady=(0, 8), sticky="w")

        self.table_text = ctk.CTkTextbox(self.sidebar, fg_color="#0f172a", corner_radius=12, font=("Courier", 12), wrap="none")
        self.table_text.grid(row=11, column=0, padx=24, pady=(0, 18), sticky="nsew")
        self.table_text.insert("0.0", "Нет активных целей\n")

        self.status_label = ctk.CTkLabel(self.sidebar, text="Готов к выбору источника", justify="left", wraplength=290, text_color="#9ca3af")
        self.status_label.grid(row=12, column=0, padx=24, pady=(0, 22), sticky="ew")

        self.header = ctk.CTkFrame(self.main, height=104, fg_color="#0b1018", corner_radius=0)
        self.header.grid(row=0, column=0, sticky="ew", padx=26, pady=(22, 8))
        self.header.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.source_card = self._make_info_block(self.header, "Источник", self.current_source, 0)
        self.mode_card = self._make_info_block(self.header, "Режим", self.current_mode, 1)
        self.fps_card = self._make_info_block(self.header, "FPS", "-", 2)
        self.resolution_card = self._make_info_block(self.header, "Кадр", self.current_resolution, 3)

        self.video_frame = ctk.CTkFrame(self.main, fg_color="#05070b", corner_radius=18)
        self.video_frame.grid(row=1, column=0, sticky="nsew", padx=26, pady=(8, 14))
        self.video_frame.grid_columnconfigure(0, weight=1)
        self.video_frame.grid_rowconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.video_frame, text="", fg_color="#05070b", text_color="#94a3b8", font=ctk.CTkFont(size=18))
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)

        self.video_label.bind("<ButtonPress-1>", self._on_mouse_press)
        self.video_label.bind("<B1-Motion>", self._on_mouse_drag)
        self.video_label.bind("<ButtonRelease-1>", self._on_mouse_release)

        self.footer = ctk.CTkFrame(self.main, height=58, fg_color="#0b1018", corner_radius=0)
        self.footer.grid(row=2, column=0, sticky="ew", padx=26, pady=(0, 20))
        self.footer.grid_columnconfigure(0, weight=1)

        self.future_label = ctk.CTkLabel(self.footer, text="Пауза, затем Z и выделите зону мышью на видео. Сброс – R. Зона следует за фоном.", text_color="#9ca3af", anchor="w")
        self.future_label.grid(row=0, column=0, sticky="ew")

    def _make_info_block(self, parent, caption, value, column):
        block = ctk.CTkFrame(parent, fg_color="#111827", corner_radius=14)
        block.grid(row=0, column=column, padx=6, pady=4, sticky="nsew")
        block.grid_columnconfigure(0, weight=1)
        caption_label = ctk.CTkLabel(block, text=caption, text_color="#8b949e", font=ctk.CTkFont(size=12), anchor="w")
        caption_label.grid(row=0, column=0, padx=16, pady=(12, 0), sticky="ew")
        value_label = ctk.CTkLabel(block, text=value, text_color="#f8fafc", font=ctk.CTkFont(size=16, weight="bold"), anchor="w")
        value_label.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="ew")
        return value_label

    def toggle_detections(self):
        self.show_detections = self.detect_switch.get()
        if not self.show_detections:
            self.tracker = KalmanTracker()
        if self.paused:
            self._redraw_paused_frame()

    # ---------- ПАУЗА ----------
    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_button.configure(text="Продолжить")
            self.status_label.configure(text="Пауза. Выделите зону мышкой (Z).")
            if self.frame_job is not None:
                self.after_cancel(self.frame_job)
                self.frame_job = None
        else:
            self.pause_button.configure(text="Пауза")
            self.status_label.configure(text="Воспроизведение возобновлено.")
            if self.capture is not None:
                self._schedule_frame()

    # ---------- ЗОНА ИНТЕРЕСА ----------
    def enable_roi_selection(self):
        self.selecting_roi = True
        self.roi_start = None
        self.status_label.configure(text="Режим выделения зоны. Обведите мышью на видео.")

    def reset_roi(self):
        self.roi = None
        self.roi_anchors = None
        self.roi_anchor_frame = None
        self.selecting_roi = False
        self.roi_start = None
        self.status_label.configure(text="Зона интереса сброшена.")
        if self.paused:
            self._redraw_paused_frame()

    def _on_mouse_press(self, event):
        if not self.selecting_roi:
            return
        self.roi_start = (event.x, event.y)

    def _on_mouse_drag(self, event):
        pass

    def _on_mouse_release(self, event):
        if not self.selecting_roi or self.roi_start is None:
            return
        x1, y1 = self.roi_start
        x2, y2 = event.x, event.y

        if self.capture is not None:
            label_w = self.video_label.winfo_width()
            label_h = self.video_label.winfo_height()
            if label_w < 10 or label_h < 10:
                return
            frame_w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frame_w == 0 or frame_h == 0:
                return
            scale_x = frame_w / label_w
            scale_y = frame_h / label_h

            real_x1 = int(min(x1, x2) * scale_x)
            real_y1 = int(min(y1, y2) * scale_y)
            real_x2 = int(max(x1, x2) * scale_x)
            real_y2 = int(max(y1, y2) * scale_y)

            self.roi = (real_x1, real_y1, real_x2, real_y2)

            if self.last_gray is not None:
                self.roi_anchors = self._find_roi_anchors(self.last_gray)
                self.roi_anchor_frame = self.last_gray.copy()
            else:
                self.roi_anchors = None
                self.roi_anchor_frame = None

            self.status_label.configure(text=f"Зона задана. Якорей: {len(self.roi_anchors) if self.roi_anchors is not None else 0}")
            self.selecting_roi = False

            if self.paused:
                self._redraw_paused_frame()

    def _find_roi_anchors(self, gray_frame):
        if self.roi is None:
            return None
        rx1, ry1, rx2, ry2 = self.roi
        mask = np.zeros_like(gray_frame)
        cv2.rectangle(mask, (rx1, ry1), (rx2, ry2), 255, -1)
        corners = cv2.goodFeaturesToTrack(gray_frame, maxCorners=20, qualityLevel=0.01, minDistance=10, mask=mask)
        if corners is None:
            return None
        return corners.reshape(-1, 2)

    def _update_roi_position(self, new_gray):
        if self.roi is None or self.roi_anchors is None or self.roi_anchor_frame is None:
            return
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.roi_anchor_frame, new_gray,
            self.roi_anchors.reshape(-1, 1, 2).astype(np.float32), None)
        if new_points is None or status is None:
            return
        good_old = self.roi_anchors[status.flatten() == 1]
        good_new = new_points[status.flatten() == 1].reshape(-1, 2)
        if len(good_old) < 4:
            self.roi_anchors = None
            self.roi_anchor_frame = None
            return
        dx = np.mean(good_new[:, 0] - good_old[:, 0])
        dy = np.mean(good_new[:, 1] - good_old[:, 1])
        rx1, ry1, rx2, ry2 = self.roi
        self.roi = (int(rx1 + dx), int(ry1 + dy), int(rx2 + dx), int(ry2 + dy))
        self.roi_anchors = good_new
        self.roi_anchor_frame = new_gray

    # ========== ИСТОЧНИКИ ВИДЕО ==========
    def show_empty_view(self):
        self.video_label.configure(image=None, text="Выбери камеру или загрузи видеофайл")
        self._update_header(source="Источник не выбран", mode="Ожидание", fps="-", resolution="-")

    def scan_cameras(self):
        if self.scanning:
            return
        self.stop_source(clear_screen=False)
        self.scanning = True
        self.camera_button.configure(state="disabled", text="Ищу камеры...")
        self.status_label.configure(text="Проверяю подключенные камеры...")
        self._clear_camera_list()
        self._add_camera_placeholder("Поиск камер...")
        thread = threading.Thread(target=self._scan_cameras_worker, daemon=True)
        thread.start()

    def _scan_cameras_worker(self):
        cameras: list[CameraInfo] = []
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        for index in range(10):
            capture = cv2.VideoCapture(index, backend)
            if not capture.isOpened():
                capture.release()
                continue
            ok, frame = capture.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
                cameras.append(CameraInfo(index=index, width=width, height=height))
            capture.release()
        self.after(0, lambda: self._show_camera_results(cameras))

    def _show_camera_results(self, cameras: list[CameraInfo]):
        self.scanning = False
        self.camera_button.configure(state="normal", text="Работать с камерой")
        self._clear_camera_list()
        if not cameras:
            self._add_camera_placeholder("Камеры не найдены")
            self.status_label.configure(text="Не удалось найти доступные камеры.")
            return
        self.status_label.configure(text=f"Найдено камер: {len(cameras)}.")
        for row, camera in enumerate(cameras):
            button = ctk.CTkButton(self.camera_list, text=f"{camera.title}\n{camera.details}", height=62, corner_radius=10, fg_color="#1f2937", hover_color="#334155", anchor="w", command=lambda item=camera: self.open_camera(item))
            button.grid(row=row, column=0, padx=8, pady=(8, 0), sticky="ew")

    def open_camera(self, camera: CameraInfo):
        self.stop_source(clear_screen=False)
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        capture = cv2.VideoCapture(camera.index, backend)
        if not capture.isOpened():
            messagebox.showerror("Камера не открылась", f"Не удалось открыть камеру {camera.index}.")
            self.show_empty_view()
            return
        self.capture = capture
        self.current_source = camera.title
        self.current_mode = "Камера"
        self.current_resolution = camera.details
        self.video_delay_ms = 15
        self.smoothed_fps = 0.0
        self.last_frame_at = time.perf_counter()
        self.paused = False
        self.pause_button.configure(text="Пауза")
        self.status_label.configure(text=f"Открыта {camera.title}. Идет детекция.")
        self._update_header(source=self.current_source, mode=self.current_mode, fps="-", resolution=self.current_resolution)
        self._schedule_frame()

    def open_video_dialog(self):
        path = filedialog.askopenfilename(title="Выбери видео для теста", filetypes=[("Видео", "*.mp4 *.avi *.mov *.mkv *.webm"), ("Все файлы", "*.*")])
        if path:
            self.open_video(Path(path))

    def open_video(self, path: Path):
        self.stop_source(clear_screen=False)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            messagebox.showerror("Видео не открылось", f"Не удалось открыть файл:\n{path}")
            self.show_empty_view()
            return
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 1 or fps > 120:
            fps = 30
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.capture = capture
        self.current_source = path.name
        self.current_mode = "Тест на видео"
        self.current_resolution = f"{width} x {height}" if width and height else "-"
        self.video_delay_ms = max(1, int(1000 / fps))
        self.smoothed_fps = 0.0
        self.last_frame_at = time.perf_counter()
        self.paused = False
        self.pause_button.configure(text="Пауза")
        self.status_label.configure(text="Видеофайл открыт. Идет детекция.")
        self._update_header(source=self.current_source, mode=self.current_mode, fps="-", resolution=self.current_resolution)
        self._schedule_frame()

    def stop_source(self, clear_screen: bool = True):
        if self.frame_job is not None:
            self.after_cancel(self.frame_job)
            self.frame_job = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.paused = False
        self.pause_button.configure(text="Пауза")
        self.tracker = KalmanTracker()
        if clear_screen:
            self.status_label.configure(text="Источник остановлен.")
            self.show_empty_view()
            self.table_text.delete("0.0", "end")
            self.table_text.insert("0.0", "Нет активных целей\n")

    def _schedule_frame(self):
        self.frame_job = self.after(self.video_delay_ms, self._read_next_frame)

    def _read_next_frame(self):
        if self.paused or self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.stop_source(clear_screen=False)
            self.video_label.configure(image=None, text="Поток завершен")
            self._update_header(fps="-")
            self.status_label.configure(text="Источник завершился.")
            return

        self.last_raw_frame = frame.copy()

        now = time.perf_counter()
        instant_fps = 1.0 / max(now - self.last_frame_at, 1e-6)
        self.last_frame_at = now
        self.smoothed_fps = instant_fps if self.smoothed_fps == 0 else self.smoothed_fps * 0.88 + instant_fps * 0.12

        if self.model is not None and self.show_detections:
            processed_frame = self._process_frame(frame)
        else:
            processed_frame = frame

        self._draw_frame(processed_frame)
        self._update_header(fps=f"{self.smoothed_fps:.1f}")
        self._schedule_frame()

    def _redraw_paused_frame(self):
        if not self.paused or self.last_raw_frame is None:
            return
        if self.show_detections and self.model is not None:
            processed = self._process_frame(self.last_raw_frame.copy())
        else:
            processed = self.last_raw_frame.copy()
            if self.roi:
                rx1, ry1, rx2, ry2 = self.roi
                cv2.rectangle(processed, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
                cv2.putText(processed, "FORBIDDEN ZONE", (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        self._draw_frame(processed)

    # ==================== ГЛАВНЫЙ КОНВЕЙЕР ====================
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)

        self.last_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not self.paused and self.roi_anchors is not None:
            self._update_roi_position(self.last_gray)

        # Предобработка
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharp = cv2.filter2D(frame, -1, sharpen_kernel)
        lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        # Основной проход без TTA
        results = self.model(enhanced, augment=False, conf=CONF_THRESHOLD, verbose=False)[0]

        detections, labels, confs = [], [], []
        for box in results.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            if label not in TARGET_CLASSES:
                continue

            cls_thresh = CLASS_THRESHOLDS.get(label, CONF_THRESHOLD)
            if area < 5000:
                cls_thresh -= 0.1
            if conf < cls_thresh:
                continue

            detections.append([x1, y1, x2, y2])
            labels.append(label)
            confs.append(conf)

        # Адаптивный проход с TTA
        if len(detections) == 0:
            results_low = self.model(enhanced, augment=True, conf=CONF_LOW, verbose=False)[0]
            for box in results_low.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                if label not in TARGET_CLASSES:
                    continue

                effective_thresh = CONF_LOW - 0.1 if area < 5000 else CONF_LOW
                if conf < effective_thresh:
                    continue

                detections.append([x1, y1, x2, y2])
                labels.append(label)
                confs.append(conf)

        # ----- Коррекция конфликтов drone vs helicopter -----
        for i in range(len(detections)):
            if labels[i] != "drone":
                continue
            for j in range(len(detections)):
                if i == j or labels[j] != "helicopter":
                    continue
                xA = max(detections[i][0], detections[j][0])
                yA = max(detections[i][1], detections[j][1])
                xB = min(detections[i][2], detections[j][2])
                yB = min(detections[i][3], detections[j][3])
                inter = max(0, xB - xA) * max(0, yB - yA)
                if inter == 0:
                    continue
                area_i = (detections[i][2] - detections[i][0]) * (detections[i][3] - detections[i][1])
                area_j = (detections[j][2] - detections[j][0]) * (detections[j][3] - detections[j][1])
                iou = inter / (area_i + area_j - inter + 1e-6)
                if iou > 0.4:
                    if confs[j] >= 0.25:          # вертолёт хоть как-то виден
                        confs[i] *= 0.7           # ослабляем дрон
                        confs[j] = min(confs[j] * 1.2, 1.0)   # немного поднимаем вертолёт

        # ----- Фильтрация дубликатов только одинаковых классов -----
        if len(detections) > 1:
            to_remove = set()
            for i in range(len(detections)):
                for j in range(i + 1, len(detections)):
                    if j in to_remove or i in to_remove:
                        continue
                    if labels[i] != labels[j]:
                        continue
                    xA = max(detections[i][0], detections[j][0])
                    yA = max(detections[i][1], detections[j][1])
                    xB = min(detections[i][2], detections[j][2])
                    yB = min(detections[i][3], detections[j][3])
                    inter = max(0, xB - xA) * max(0, yB - yA)
                    area_i = (detections[i][2] - detections[i][0]) * (detections[i][3] - detections[i][1])
                    area_j = (detections[j][2] - detections[j][0]) * (detections[j][3] - detections[j][1])
                    iou = inter / (area_i + area_j - inter + 1e-6)
                    if iou > 0.5:
                        if confs[i] < confs[j]:
                            to_remove.add(i)
                        else:
                            to_remove.add(j)
            for idx in sorted(to_remove, reverse=True):
                del detections[idx]
                del labels[idx]
                del confs[idx]
        
        self.last_detection_count = len(detections)
        tracks_dict = self.tracker.update(detections, labels, confidences=confs)

        scored = []
        for tid, tr in tracks_dict.items():
            if not tr.centers or not tr.confirmed:
                continue
            if tr.unseen > 5:
                continue
            threat = compute_threat(tr, center, self.roi)
            scored.append((tid, tr, threat))
        scored = sorted(scored, key=lambda x: x[2], reverse=True)[:MAX_OBJECTS]

        self._update_table(scored)

        if self.roi:
            rx1, ry1, rx2, ry2 = self.roi
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
            cv2.putText(frame, "FORBIDDEN ZONE", (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for rank, (tid, tr, threat) in enumerate(scored):
            if not tr.confirmed:
                continue
            if tr.unseen > 5:
                continue
            x1, y1, x2, y2 = tr.bbox
            cx, cy = tr.centroid

            in_zone = False
            if self.roi:
                rx1, ry1, rx2, ry2 = self.roi
                in_zone = rx1 < cx < rx2 and ry1 < cy < ry2

            color = (0, 0, 255) if (rank == 0 or in_zone) else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            display_conf = tr.avg_conf
            conf_str = f" {display_conf:.2f}" if display_conf > 0 else ""
            label_text = f"ID {tid} [{tr.label}]{conf_str}"
            if in_zone:
                label_text += " ALARM!"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"({cx},{cy})", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            for i in range(1, len(tr.centers)):
                cv2.line(frame, tr.centers[i - 1], tr.centers[i], (255, 0, 0), 2)

            pred = tr.predict()
            if pred:
                px, py = pred
                cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)

        return frame

    def _update_table(self, scored):
        if self.table_text is None:
            return
        self.table_text.delete("0.0", "end")
        if not scored:
            self.table_text.insert("0.0", "Нет активных целей\n")
            return
        header = f"{'ID':<4} {'Тип':<12} {'X':<6} {'Y':<6} {'Угроза':<8}\n"
        self.table_text.insert("0.0", header)
        for tid, tr, threat in scored:
            cx, cy = tr.centroid
            line = f"{tid:<4} {tr.label:<12} {cx:<6} {cy:<6} {int(threat):<8}\n"
            self.table_text.insert("end", line)

    def _draw_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        label_width = max(self.video_label.winfo_width(), 640)
        label_height = max(self.video_label.winfo_height(), 420)
        image.thumbnail((label_width, label_height), Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self.current_photo, text="")

    def _update_header(self, source=None, mode=None, fps=None, resolution=None):
        if source is not None:
            self.source_card.configure(text=source)
        if mode is not None:
            self.mode_card.configure(text=mode)
        if fps is not None:
            self.fps_card.configure(text=fps)
        if resolution is not None:
            self.resolution_card.configure(text=resolution)

    def _clear_camera_list(self):
        for widget in self.camera_list.winfo_children():
            widget.destroy()

    def _add_camera_placeholder(self, text):
        label = ctk.CTkLabel(self.camera_list, text=text, text_color="#9ca3af", height=52, anchor="w")
        label.grid(row=0, column=0, padx=12, pady=12, sticky="ew")

    def close_app(self):
        self.stop_source(clear_screen=False)
        self.destroy()


if __name__ == "__main__":
    app = BurevestnikSystem()
    app.mainloop()
