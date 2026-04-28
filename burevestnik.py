from __future__ import annotations
import os, threading, time, glob
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox
import cv2, customtkinter as ctk, numpy as np
from PIL import Image, ImageTk
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
ALARM_SOUND_PATH = PROJECT_ROOT / "ресурсы" / "тревога.wav"
ALARM_SOUND_COOLDOWN_SECONDS = 2.0
ALERT_MEMORY_SECONDS = 12.0
ALERT_SAME_OBJECT_DISTANCE = 80.0

MAX_OBJECTS = 5
CONF_THRESHOLD = 0.4
CONF_LOW = 0.3
TARGET_CLASSES = ["drone", "airplane", "helicopter"]
MAX_TRACK_AGE = 60
TRACK_DIST_THRESH = 0.7
CLASS_THRESHOLDS = {"drone": 0.45, "airplane": 0.4, "helicopter": 0.35}

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
        self.label_history = deque(maxlen=10)
        self.label_history.append(label)
        self.pred_bbox = list(bbox)
        #Держим простую модель движения, чтобы не терять цель между кадрами.
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.kf.P *= 10.0
        self.kf.R = np.diag([5.0, 5.0])
        self.kf.Q = np.diag([1.0, 1.0, 1.0, 1.0])
        x1,y1,x2,y2 = bbox
        cx = (x1+x2)/2.0
        cy = (y1+y2)/2.0
        self.kf.x = np.array([cx,cy,0.0,0.0])
        self.centroid = (int(cx), int(cy))
        self.centers.append(self.centroid)
    def predict(self):
        self.kf.predict()
        pred_x, pred_y = self.kf.x[0], self.kf.x[1]
        w = self.pred_bbox[2] - self.pred_bbox[0]
        h = self.pred_bbox[3] - self.pred_bbox[1]
        self.pred_bbox = [int(pred_x - w/2), int(pred_y - h/2), int(pred_x + w/2), int(pred_y + h/2)]
        return int(pred_x), int(pred_y)
    def update(self, bbox, label, conf=0.0):
        self.label_history.append(label)
        x1,y1,x2,y2 = bbox
        cx = (x1+x2)/2.0
        cy = (y1+y2)/2.0
        if len(self.centers) == 1 and not self.confirmed:
            old_cx, old_cy = self.centers[0]
            self.kf.x[2] = cx - old_cx
            self.kf.x[3] = cy - old_cy
        self.kf.update(np.array([cx,cy]))
        self.bbox = bbox
        self.label = label
        self.conf = conf
        self.conf_history.append(conf)
        self.pred_bbox = list(bbox)
        self.centroid = (int(self.kf.x[0]), int(self.kf.x[1]))
        self.centers.append(self.centroid)
        self.unseen = 0
        self.age += 1
        self.hits += 1
        if self.hits >= 3: self.confirmed = True
    def mark_missed(self):
        self.unseen += 1
        self.kf.predict()
        self.centroid = (int(self.kf.x[0]), int(self.kf.x[1]))
        w = self.pred_bbox[2] - self.pred_bbox[0]
        h = self.pred_bbox[3] - self.pred_bbox[1]
        self.pred_bbox = [int(self.centroid[0] - w/2), int(self.centroid[1] - h/2),
                          int(self.centroid[0] + w/2), int(self.centroid[1] + h/2)]
    def is_dead(self): return self.unseen > MAX_TRACK_AGE
    @property
    def velocity(self): return self.kf.x[2], self.kf.x[3]
    @property
    def avg_conf(self):
        if not self.conf_history: return 0.0
        return sum(self.conf_history)/len(self.conf_history)
    @property
    def best_label(self):
        if not self.label_history: return self.label
        return max(set(self.label_history), key=self.label_history.count)

class KalmanTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1
    def update(self, detections, labels, confidences=None):
        for tr in self.tracks: tr.predict()
        #Считаем, насколько каждая новая рамка похожа на уже существующий трек.
        cost = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for i, tr in enumerate(self.tracks):
            pred_bbox = tr.pred_bbox
            pred_center = ((pred_bbox[0]+pred_bbox[2])//2, (pred_bbox[1]+pred_bbox[3])//2)
            pw = pred_bbox[2]-pred_bbox[0]
            ph = pred_bbox[3]-pred_bbox[1]
            for j, det in enumerate(detections):
                dc = ((det[0]+det[2])//2, (det[1]+det[3])//2)
                dist = np.linalg.norm(np.array(pred_center)-np.array(dc))
                max_dim = max(pw, ph, det[2]-det[0], det[3]-det[1]) + 1e-6
                dist_norm = dist/max_dim
                xA = max(pred_bbox[0], det[0])
                yA = max(pred_bbox[1], det[1])
                xB = min(pred_bbox[2], det[2])
                yB = min(pred_bbox[3], det[3])
                inter = max(0, xB-xA)*max(0, yB-yA)
                area_tr = (pred_bbox[2]-pred_bbox[0])*(pred_bbox[3]-pred_bbox[1])
                area_det = (det[2]-det[0])*(det[3]-det[1])
                iou = inter/(area_tr+area_det-inter+1e-6)
                cost[i,j] = 0.6*dist_norm + 0.4*(1.0-iou)
                if confidences and confidences[j]>0: cost[i,j] = cost[i,j]/(confidences[j]+0.1)
        #Подбираем лучшее соответствие "старый трек -> новая детекция".
        row_ind, col_ind = linear_sum_assignment(cost)
        assigned_t, assigned_d = set(), set()
        for r,c in zip(row_ind, col_ind):
            if cost[r,c] < TRACK_DIST_THRESH:
                self.tracks[r].update(detections[c], labels[c], confidences[c] if confidences else 0.0)
                assigned_t.add(r); assigned_d.add(c)
        for i, tr in enumerate(self.tracks):
            if i not in assigned_t: tr.mark_missed()
        for j, det in enumerate(detections):
            if j not in assigned_d:
                nt = Track(self.next_id, det, labels[j], confidences[j] if confidences else 0.0)
                self.tracks.append(nt)
                self.next_id += 1
        self.tracks = [tr for tr in self.tracks if not tr.is_dead()]
        return {tr.id: tr for tr in self.tracks}

def compute_threat(track, frame_center, roi=None):
    #Угроза растет, если цель ближе к центру, быстрее движется или входит в запретную зону.
    cx,cy = track.centroid
    dist = np.hypot(cx-frame_center[0], cy-frame_center[1])
    vx,vy = track.velocity
    speed = np.hypot(vx,vy)
    threat = (1000-dist)*0.5 + speed*10
    if roi is not None:
        rx1,ry1,rx2,ry2 = roi
        if rx1<cx<rx2 and ry1<cy<ry2: threat += 500
    if track.best_label in ("airplane","helicopter"): threat += 200
    elif track.best_label == "drone": threat += 400
    elif track.best_label == "bird": threat -= 150
    return threat

@dataclass(frozen=True)
class CameraInfo:
    index: int; width: int; height: int
    @property
    def title(self): return f"Камера {self.index}"
    @property
    def details(self): return f"{self.width} x {self.height}"

class AirSpaceMonitor(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("AirSpace Monitor – обнаружение и сопровождение")
        self.geometry("1280x800")
        self.minsize(1060,680)
        self.configure(fg_color="#0b1018")
        self.capture = None
        self.frame_job = None
        self.current_photo = None
        self.current_pil_image = None
        self.current_source = "Источник не выбран"
        self.current_mode = "Ожидание"
        self.current_resolution = "-"
        self.video_delay_ms = 15
        self.last_frame_at = time.perf_counter()
        self.smoothed_fps = 0.0
        self.scanning = False
        self.paused = False
        self.model = None
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
        self.model_files = []
        self.selected_model_path = "best.pt"
        self.analysis_frame_step = 1
        self.frame_counter = 0
        self.last_analysis_ms = 0.0
        self.alarmed_track_ids = set()
        self.alert_memory = {}
        self.alert_reset_job = None
        self.last_alarm_sound_at = 0.0
        self.bind("<KeyPress-z>", lambda e: self.enable_roi_selection())
        self.bind("<KeyPress-r>", lambda e: self.reset_roi())
        self.after(100, self._load_initial_model)
        self.table_text = None
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        self._build_layout()
        self.show_empty_view()

    def _load_initial_model(self):
        #При старте подхватываем первую найденную модель, чтобы приложение сразу было готово к работе.
        self._scan_model_files()
        if self.model_files:
            self.selected_model_path = self.model_files[0]
            self.model_selector.set(os.path.basename(self.selected_model_path))
        self._load_model()

    def _load_model(self):
        try:
            self.status_label.configure(text="Загружаю модель...")
            self.model = YOLO(self.selected_model_path)
            self.status_label.configure(text="Модель загружена. Готов к выбору источника.")
        except Exception as e:
            self.status_label.configure(text=f"Ошибка загрузки модели: {e}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель YOLO:\n{e}")

    def _scan_model_files(self):
        self.model_files = glob.glob("*.pt")
        if not self.model_files:
            self.model_files = ["best.pt"]

    def _on_model_select(self, choice):
        for path in self.model_files:
            if os.path.basename(path) == choice:
                self.selected_model_path = path
                break
        self.tracker = KalmanTracker()
        self.alarmed_track_ids.clear()
        self.alert_memory.clear()
        self._load_model()
        self.status_label.configure(text=f"Модель {choice} загружена.")

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Выберите файл модели",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")]
        )
        if path:
            self.selected_model_path = path
            self.tracker = KalmanTracker()
            self.alarmed_track_ids.clear()
            self.alert_memory.clear()
            self._load_model()
            self.status_label.configure(text=f"Загружена модель {os.path.basename(path)}")
            if path not in self.model_files:
                self.model_files.append(path)
                self.model_selector.configure(values=[os.path.basename(p) for p in self.model_files])
            self.model_selector.set(os.path.basename(path))

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=350, corner_radius=0, fg_color="#111827")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(9, weight=1)

        t = ctk.CTkLabel(self.sidebar, text="AirSpace Monitor", font=ctk.CTkFont(size=24, weight="bold"), text_color="#f8fafc")
        t.grid(row=0, column=0, padx=24, pady=(28,0), sticky="w")
        st = ctk.CTkLabel(self.sidebar, text="Обнаружение и сопровождение\nвоздушных объектов", justify="left", font=ctk.CTkFont(size=14), text_color="#9ca3af")
        st.grid(row=1, column=0, padx=24, pady=(4,12), sticky="w")

        model_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        model_frame.grid(row=2, column=0, padx=24, pady=(0,12), sticky="ew")
        model_frame.grid_columnconfigure(0, weight=1)
        model_frame.grid_columnconfigure(1, weight=0)

        self.model_selector = ctk.CTkOptionMenu(
            model_frame,
            values=["best.pt"],
            command=self._on_model_select,
            fg_color="#1f2937",
            button_color="#3b82f6",
            dropdown_fg_color="#1f2937"
        )
        self.model_selector.grid(row=0, column=0, padx=(0,6), sticky="ew")
        browse_btn = ctk.CTkButton(
            model_frame,
            text="Обзор",
            width=60,
            height=32,
            corner_radius=8,
            fg_color="#374151",
            hover_color="#4b5563",
            command=self._browse_model
        )
        browse_btn.grid(row=0, column=1)

        analysis_frame = ctk.CTkFrame(self.sidebar, fg_color="#0f172a", corner_radius=12)
        analysis_frame.grid(row=3, column=0, padx=24, pady=(0,12), sticky="ew")
        analysis_frame.grid_columnconfigure(0, weight=1)
        analysis_frame.grid_columnconfigure(1, weight=0)
        analysis_title = ctk.CTkLabel(analysis_frame, text="Кадры между анализами", text_color="#e5e7eb", font=ctk.CTkFont(size=13, weight="bold"), anchor="w")
        analysis_title.grid(row=0, column=0, padx=12, pady=(10,0), sticky="ew")
        self.analysis_step_value_label = ctk.CTkLabel(analysis_frame, text="1", text_color="#f8fafc", font=ctk.CTkFont(size=13, weight="bold"), anchor="e")
        self.analysis_step_value_label.grid(row=0, column=1, padx=12, pady=(10,0), sticky="e")
        self.analysis_step_slider = ctk.CTkSlider(
            analysis_frame,
            from_=1,
            to=30,
            number_of_steps=29,
            command=self._on_analysis_step_change,
        )
        self.analysis_step_slider.grid(row=1, column=0, columnspan=2, padx=12, pady=(8,8), sticky="ew")
        self.analysis_step_slider.set(self.analysis_frame_step)
        self.analysis_info_label = ctk.CTkLabel(analysis_frame, text="YOLO: -", text_color="#9ca3af", anchor="w")
        self.analysis_info_label.grid(row=2, column=0, columnspan=2, padx=12, pady=(0,10), sticky="ew")

        self.camera_button = ctk.CTkButton(self.sidebar, text="Работать с камерой", height=46, corner_radius=10, fg_color="#0ea5e9", hover_color="#0284c7", font=ctk.CTkFont(size=15, weight="bold"), command=self.scan_cameras)
        self.camera_button.grid(row=4, column=0, padx=24, pady=(0,12), sticky="ew")
        self.video_button = ctk.CTkButton(self.sidebar, text="Тест на видео", height=46, corner_radius=10, fg_color="#14b8a6", hover_color="#0f766e", font=ctk.CTkFont(size=15, weight="bold"), command=self.open_video_dialog)
        self.video_button.grid(row=5, column=0, padx=24, pady=(0,12), sticky="ew")
        self.pause_button = ctk.CTkButton(self.sidebar, text="Пауза", height=40, corner_radius=10, fg_color="#f59e0b", hover_color="#d97706", command=self.toggle_pause)
        self.pause_button.grid(row=6, column=0, padx=24, pady=(0,6), sticky="ew")
        # Кнопка "Остановить" удалена
        self.detect_switch = ctk.CTkSwitch(self.sidebar, text="Показывать объекты", command=self.toggle_detections, onvalue=True, offvalue=False)
        self.detect_switch.select()
        self.detect_switch.grid(row=7, column=0, padx=24, pady=(0,12), sticky="w")

        rf = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        rf.grid(row=8, column=0, padx=24, pady=(0,12), sticky="ew")
        rf.grid_columnconfigure(0, weight=1)
        rf.grid_columnconfigure(1, weight=1)
        self.roi_button = ctk.CTkButton(rf, text="Задать зону (Z)", height=36, corner_radius=8, fg_color="#3b82f6", hover_color="#2563eb", command=self.enable_roi_selection)
        self.roi_button.grid(row=0, column=0, padx=(0,6), sticky="ew")
        self.roi_reset_button = ctk.CTkButton(rf, text="Сбросить (R)", height=36, corner_radius=8, fg_color="#4b5563", hover_color="#6b7280", command=self.reset_roi)
        self.roi_reset_button.grid(row=0, column=1, padx=(6,0), sticky="ew")

        ct = ctk.CTkLabel(self.sidebar, text="Найденные камеры", font=ctk.CTkFont(size=16, weight="bold"), text_color="#e5e7eb")
        ct.grid(row=9, column=0, padx=24, pady=(8,8), sticky="w")

        self.camera_list = ctk.CTkScrollableFrame(self.sidebar, fg_color="#0f172a", corner_radius=12)
        self.camera_list.grid(row=10, column=0, padx=24, pady=(0,12), sticky="nsew")
        self.camera_list.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(self.sidebar, text="Готов к выбору источника", justify="left", wraplength=290, text_color="#9ca3af")
        self.status_label.grid(row=11, column=0, padx=24, pady=(0,22), sticky="ew")

        # Правая часть
        self.main = ctk.CTkFrame(self, corner_radius=0, fg_color="#0b1018")
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(0, weight=0)
        self.main.grid_rowconfigure(1, weight=0)
        self.main.grid_rowconfigure(2, weight=1)
        self.main.grid_rowconfigure(3, weight=0)
        self.main.grid_rowconfigure(4, weight=0)

        self.header = ctk.CTkFrame(self.main, height=104, fg_color="#0b1018", corner_radius=0)
        self.header.grid(row=0, column=0, sticky="ew", padx=26, pady=(22,8))
        self.header.grid_columnconfigure((0,1,2,3), weight=1)
        self.source_card = self._make_info_block(self.header, "Источник", self.current_source, 0)
        self.mode_card = self._make_info_block(self.header, "Режим", self.current_mode, 1)
        self.fps_card = self._make_info_block(self.header, "FPS", "-", 2)
        self.resolution_card = self._make_info_block(self.header, "Кадр", self.current_resolution, 3)

        self.alert_label = ctk.CTkLabel(
            self.main,
            text="Оповещение: новых объектов нет",
            height=34,
            fg_color="#111827",
            corner_radius=10,
            text_color="#94a3b8",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        )
        self.alert_label.grid(row=1, column=0, sticky="ew", padx=26, pady=(0,6))

        self.video_frame = ctk.CTkFrame(self.main, fg_color="#05070b", corner_radius=18)
        self.video_frame.grid(row=2, column=0, sticky="nsew", padx=26, pady=(8,8))
        self.video_frame.grid_columnconfigure(0, weight=1)
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_label = ctk.CTkLabel(self.video_frame, text="", fg_color="#05070b", text_color="#94a3b8", font=ctk.CTkFont(size=18))
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)

        # Таблица под видео
        table_frame = ctk.CTkFrame(self.main, fg_color="#0b1018", corner_radius=0)
        table_frame.grid(row=3, column=0, sticky="ew", padx=26, pady=(0,8))
        table_frame.grid_columnconfigure(0, weight=1)
        self.table_text = ctk.CTkTextbox(table_frame, fg_color="#111827", corner_radius=12, font=("Courier",12), wrap="none", height=120)
        self.table_text.grid(row=0, column=0, sticky="ew")
        self.table_text.insert("0.0", "Нет активных целей\n")

        self.footer = ctk.CTkFrame(self.main, height=58, fg_color="#0b1018", corner_radius=0)
        self.footer.grid(row=4, column=0, sticky="ew", padx=26, pady=(0,20))
        self.footer.grid_columnconfigure(0, weight=1)
        self.future_label = ctk.CTkLabel(self.footer, text="Пауза, затем Z и выделите зону мышью на видео. Сброс – R. Зона следует за фоном.", text_color="#9ca3af", anchor="w")
        self.future_label.grid(row=0, column=0, sticky="ew")

        self.video_label.bind("<ButtonPress-1>", self._on_mouse_press)
        self.video_label.bind("<B1-Motion>", self._on_mouse_drag)
        self.video_label.bind("<ButtonRelease-1>", self._on_mouse_release)

    def _make_info_block(self, parent, caption, value, column):
        b = ctk.CTkFrame(parent, fg_color="#111827", corner_radius=14)
        b.grid(row=0, column=column, padx=6, pady=4, sticky="nsew")
        b.grid_columnconfigure(0, weight=1)
        cl = ctk.CTkLabel(b, text=caption, text_color="#8b949e", font=ctk.CTkFont(size=12), anchor="w")
        cl.grid(row=0, column=0, padx=16, pady=(12,0), sticky="ew")
        vl = ctk.CTkLabel(b, text=value, text_color="#f8fafc", font=ctk.CTkFont(size=16, weight="bold"), anchor="w")
        vl.grid(row=1, column=0, padx=16, pady=(0,12), sticky="ew")
        return vl

    def toggle_detections(self):
        self.show_detections = self.detect_switch.get()
        if not self.show_detections:
            self.tracker = KalmanTracker()
            self.alarmed_track_ids.clear()
            self.alert_memory.clear()
        if self.paused: self._redraw_paused_frame()

    def _on_analysis_step_change(self, value):
        self.analysis_frame_step = max(1, int(round(float(value))))
        self.analysis_step_value_label.configure(text=str(self.analysis_frame_step))
        self._update_analysis_info()

    def _reset_analysis_state(self):
        self.frame_counter = 0
        self.last_analysis_ms = 0.0
        self._update_analysis_info()

    def _should_analyze_current_frame(self):
        return self.frame_counter == 1 or self.frame_counter % self.analysis_frame_step == 0

    def _update_analysis_info(self):
        if not hasattr(self, "analysis_info_label"):
            return
        self.analysis_info_label.configure(
            text=f"Шаг: {self.analysis_frame_step} кадр. | YOLO: {self.last_analysis_ms:.0f} мс"
        )

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
            if self.capture is not None: self._schedule_frame()

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
        if self.paused: self._redraw_paused_frame()

    def _on_mouse_press(self, event):
        if not self.selecting_roi: return
        self.roi_start = (event.x, event.y)

    def _on_mouse_drag(self, event): pass

    def _on_mouse_release(self, event):
        if not self.selecting_roi or self.roi_start is None: return
        x1, y1 = self.roi_start
        x2, y2 = event.x, event.y
        if self.capture is not None:
            if self.current_pil_image is None: return
            lw = self.video_label.winfo_width()
            lh = self.video_label.winfo_height()
            if lw < 10 or lh < 10: return

            imw, imh = self.current_pil_image.size
            offset_x = (lw - imw) // 2
            offset_y = (lh - imh) // 2

            ix1 = x1 - offset_x
            iy1 = y1 - offset_y
            ix2 = x2 - offset_x
            iy2 = y2 - offset_y

            ix1 = max(0, min(ix1, imw - 1))
            iy1 = max(0, min(iy1, imh - 1))
            ix2 = max(0, min(ix2, imw - 1))
            iy2 = max(0, min(iy2, imh - 1))

            fw = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fw == 0 or fh == 0: return

            sx = fw / imw
            sy = fh / imh

            rx1 = int(ix1 * sx)
            ry1 = int(iy1 * sy)
            rx2 = int(ix2 * sx)
            ry2 = int(iy2 * sy)

            self.roi = (rx1, ry1, rx2, ry2)
            if self.last_gray is not None:
                self.roi_anchors = self._find_roi_anchors(self.last_gray)
                self.roi_anchor_frame = self.last_gray.copy()
            else:
                self.roi_anchors = None
                self.roi_anchor_frame = None
            ac = len(self.roi_anchors) if self.roi_anchors is not None else 0
            self.status_label.configure(text=f"Зона задана. Якорей: {ac}")
            self.selecting_roi = False
            if self.paused: self._redraw_paused_frame()

    def _find_roi_anchors(self, gray_frame):
        if self.roi is None: return None
        #Запоминаем заметные точки внутри зоны, чтобы потом сдвигать ее вместе с картинкой.
        rx1,ry1,rx2,ry2 = self.roi
        mask = np.zeros_like(gray_frame)
        cv2.rectangle(mask, (rx1,ry1), (rx2,ry2), 255, -1)
        corners = cv2.goodFeaturesToTrack(gray_frame, maxCorners=20, qualityLevel=0.01, minDistance=10, mask=mask)
        if corners is None: return None
        return corners.reshape(-1,2)

    def _update_roi_position(self, new_gray):
        if self.roi is None or self.roi_anchors is None or self.roi_anchor_frame is None: return
        #Сдвигаем ROI по оптическому потоку, чтобы зона не "отставала" от фона.
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.roi_anchor_frame, new_gray,
            self.roi_anchors.reshape(-1,1,2).astype(np.float32), None)
        if new_pts is None or status is None: return
        good_old = self.roi_anchors[status.flatten()==1]
        good_new = new_pts[status.flatten()==1].reshape(-1,2)
        if len(good_old)<4:
            self.roi_anchors = None
            self.roi_anchor_frame = None
            return
        dx = np.mean(good_new[:,0]-good_old[:,0])
        dy = np.mean(good_new[:,1]-good_old[:,1])
        rx1,ry1,rx2,ry2 = self.roi
        self.roi = (int(rx1+dx), int(ry1+dy), int(rx2+dx), int(ry2+dy))
        self.roi_anchors = good_new
        self.roi_anchor_frame = new_gray

    def show_empty_view(self):
        self.video_label.configure(image=None, text="Выбери камеру или загрузи видеофайл")
        self._update_header(source="Источник не выбран", mode="Ожидание", fps="-", resolution="-")

    def scan_cameras(self):
        if self.scanning: return
        self.stop_source(clear_screen=False)
        self.scanning = True
        self.camera_button.configure(state="disabled", text="Ищу камеры...")
        self.status_label.configure(text="Проверяю подключенные камеры...")
        self._clear_camera_list()
        self._add_camera_placeholder("Поиск камер...")
        threading.Thread(target=self._scan_cameras_worker, daemon=True).start()

    def _scan_cameras_worker(self):
        cameras = []
        #На Windows одни камеры лучше открываются через MSMF, другие через DirectShow.
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW] if os.name == "nt" else [cv2.CAP_ANY]
        for index in range(10):
            for backend in backends:
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        h, w = frame.shape[:2]
                        cameras.append(CameraInfo(index=index, width=w, height=h))
                        cap.release()
                        break
                    cap.release()
        self.after(0, self._show_camera_results, cameras)

    def _show_camera_results(self, cameras):
        self.scanning = False
        self.camera_button.configure(state="normal", text="Работать с камерой")
        self._clear_camera_list()
        if not cameras:
            self._add_camera_placeholder("Камеры не найдены")
            self.status_label.configure(text="Не удалось найти доступные камеры.")
            return
        self.status_label.configure(text=f"Найдено камер: {len(cameras)}.")
        for row, cam in enumerate(cameras):
            btn = ctk.CTkButton(
                self.camera_list,
                text=f"{cam.title}\n{cam.details}",
                height=62,
                corner_radius=10,
                fg_color="#1f2937",
                hover_color="#334155",
                anchor="w",
                command=lambda c=cam: self.open_camera(c)
            )
            btn.grid(row=row, column=0, padx=8, pady=(8,0), sticky="ew")
            btn.update()
        self.camera_list.update()
        self.camera_list.update_idletasks()

    def open_camera(self, camera):
        self.stop_source(clear_screen=False)
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW] if os.name == "nt" else [cv2.CAP_ANY]
        cap = None
        for backend in backends:
            cap = cv2.VideoCapture(camera.index, backend)
            if cap.isOpened():
                break
            cap.release()
        else:
            messagebox.showerror("Камера не открылась", f"Не удалось открыть камеру {camera.index}.")
            self.show_empty_view()
            return
        self.capture = cap
        self.current_source = camera.title
        self.current_mode = "Камера"
        self.current_resolution = camera.details
        self.video_delay_ms = 15
        self.smoothed_fps = 0.0
        self.last_frame_at = time.perf_counter()
        self._reset_analysis_state()
        self.paused = False
        self.pause_button.configure(text="Пауза")
        self.status_label.configure(text=f"Открыта {camera.title}. Идет детекция.")
        self._update_header(source=self.current_source, mode=self.current_mode, fps="-", resolution=self.current_resolution)
        self._schedule_frame()

    def open_video_dialog(self):
        path = filedialog.askopenfilename(title="Выбери видео для теста", filetypes=[("Видео", "*.mp4 *.avi *.mov *.mkv *.webm"), ("Все файлы", "*.*")])
        if path: self.open_video(Path(path))

    def open_video(self, path):
        self.stop_source(clear_screen=False)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            messagebox.showerror("Видео не открылось", f"Не удалось открыть файл:\n{path}")
            self.show_empty_view()
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps<=1 or fps>120: fps=30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.capture = cap
        self.current_source = path.name
        self.current_mode = "Тест на видео"
        self.current_resolution = f"{w} x {h}" if w and h else "-"
        self.video_delay_ms = max(1, int(1000/fps))
        self.smoothed_fps = 0.0
        self.last_frame_at = time.perf_counter()
        self._reset_analysis_state()
        self.paused = False
        self.pause_button.configure(text="Пауза")
        self.status_label.configure(text="Видеофайл открыт. Идет детекция.")
        self._update_header(source=self.current_source, mode=self.current_mode, fps="-", resolution=self.current_resolution)
        self._schedule_frame()

    def stop_source(self, clear_screen=True):
        if self.frame_job is not None:
            self.after_cancel(self.frame_job)
            self.frame_job = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.paused = False
        self.pause_button.configure(text="Пауза")
        self.tracker = KalmanTracker()
        self.alarmed_track_ids.clear()
        self.alert_memory.clear()
        self._reset_analysis_state()
        self._reset_alert_banner()
        self._stop_alarm_sound()
        if clear_screen:
            self.status_label.configure(text="Источник остановлен.")
            self.show_empty_view()
            self.table_text.delete("0.0","end")
            self.table_text.insert("0.0", "Нет активных целей\n")

    def _schedule_frame(self):
        self.frame_job = self.after(self.video_delay_ms, self._read_next_frame)

    def _read_next_frame(self):
        if self.paused or self.capture is None: return
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.stop_source(clear_screen=False)
            self.video_label.configure(image=None, text="Поток завершен")
            self._update_header(fps="-")
            self.status_label.configure(text="Источник завершился.")
            return
        self.frame_counter += 1
        self.last_raw_frame = frame.copy()
        now = time.perf_counter()
        instant_fps = 1.0/max(now-self.last_frame_at, 1e-6)
        self.last_frame_at = now
        self.smoothed_fps = instant_fps if self.smoothed_fps==0 else self.smoothed_fps*0.88 + instant_fps*0.12
        #Для видео можно пропускать часть кадров, иначе детекция будет слишком тяжелой.
        if self.model is not None and self.show_detections and self._should_analyze_current_frame():
            analysis_started = time.perf_counter()
            processed = self._process_frame(frame)
            self.last_analysis_ms = (time.perf_counter() - analysis_started) * 1000
        else:
            processed = frame
            if not self.show_detections:
                self._update_table([])
        self._draw_frame(processed)
        self._update_header(fps=f"{self.smoothed_fps:.1f}")
        self._update_analysis_info()
        self._schedule_frame()

    def _redraw_paused_frame(self):
        if not self.paused or self.last_raw_frame is None: return
        if self.show_detections and self.model is not None:
            processed = self._process_frame(self.last_raw_frame.copy())
        else:
            processed = self.last_raw_frame.copy()
            if self.roi:
                rx1,ry1,rx2,ry2 = self.roi
                cv2.rectangle(processed, (rx1,ry1), (rx2,ry2), (0,0,255), 2)
                cv2.putText(processed, "FORBIDDEN ZONE", (rx1,ry1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        self._draw_frame(processed)

    def _process_frame(self, frame):
        h,w = frame.shape[:2]
        center = (w//2, h//2)
        self.last_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not self.paused and self.roi_anchors is not None:
            self._update_roi_position(self.last_gray)

        #Немного подчеркиваем контраст, чтобы мелкие объекты реже терялись на небе.
        sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        sharp = cv2.filter2D(frame, -1, sharpen_kernel)
        lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)

        results = self.model(enhanced, augment=False, conf=CONF_THRESHOLD, verbose=False)[0]
        detections, labels, confs = [], [], []
        for box in results.boxes:
            conf = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            area = (x2-x1)*(y2-y1)
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            if label not in TARGET_CLASSES: continue
            cls_thresh = CLASS_THRESHOLDS.get(label, CONF_THRESHOLD)
            if area<5000: cls_thresh -= 0.1
            if conf < cls_thresh: continue
            detections.append([x1,y1,x2,y2])
            labels.append(label)
            confs.append(conf)

        #Если строгий проход ничего не дал, пробуем мягче, чтобы подхватить дальние и мелкие цели.
        if len(detections)==0:
            results_low = self.model(enhanced, augment=True, conf=CONF_LOW, verbose=False)[0]
            for box in results_low.boxes:
                conf = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                area = (x2-x1)*(y2-y1)
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                if label not in TARGET_CLASSES: continue
                eff = CONF_LOW - 0.1 if area<5000 else CONF_LOW
                if conf < eff: continue
                detections.append([x1,y1,x2,y2])
                labels.append(label)
                confs.append(conf)

        #Когда рамки дрона и вертолета сильно пересекаются, слегка смещаем уверенность в пользу вертолета.
        for i in range(len(detections)):
            if labels[i]!="drone": continue
            for j in range(len(detections)):
                if i==j or labels[j]!="helicopter": continue
                xA = max(detections[i][0], detections[j][0])
                yA = max(detections[i][1], detections[j][1])
                xB = min(detections[i][2], detections[j][2])
                yB = min(detections[i][3], detections[j][3])
                inter = max(0, xB-xA)*max(0, yB-yA)
                if inter==0: continue
                area_i = (detections[i][2]-detections[i][0])*(detections[i][3]-detections[i][1])
                area_j = (detections[j][2]-detections[j][0])*(detections[j][3]-detections[j][1])
                iou = inter/(area_i+area_j-inter+1e-6)
                if iou>0.4 and confs[j]>=0.25:
                    confs[i] *= 0.7
                    confs[j] = min(confs[j]*1.2, 1.0)

        self.last_detection_count = len(detections)
        tracks_dict = self.tracker.update(detections, labels, confidences=confs)

        scored = []
        current_zone_ids = set()
        for tid, tr in tracks_dict.items():
            if not tr.centers or not tr.confirmed: continue
            if tr.unseen>5: continue
            threat = compute_threat(tr, center, self.roi)
            scored.append((tid, tr, threat))
            if self.roi:
                cx,cy = tr.centroid
                rx1,ry1,rx2,ry2 = self.roi
                if rx1<cx<rx2 and ry1<cy<ry2:
                    current_zone_ids.add(tid)

        scored = sorted(scored, key=lambda x: x[2], reverse=True)[:MAX_OBJECTS]
        self._update_table(scored)
        self._handle_object_alerts(scored)

        if self.roi:
            rx1,ry1,rx2,ry2 = self.roi
            cv2.rectangle(frame, (rx1,ry1), (rx2,ry2), (0,0,255), 2)
            cv2.putText(frame, "FORBIDDEN ZONE", (rx1,ry1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        for rank, (tid, tr, threat) in enumerate(scored):
            if not tr.confirmed or tr.unseen>5: continue
            x1,y1,x2,y2 = tr.bbox
            cx,cy = tr.centroid
            in_zone = tid in current_zone_ids
            color = (0,0,255) if (rank==0 or in_zone) else (0,255,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            display_conf = tr.avg_conf
            conf_str = f" {display_conf:.2f}" if display_conf>0 else ""
            label_text = f"ID {tid} [{tr.best_label}]{conf_str}"
            if in_zone: label_text += " ALARM!"
            cv2.putText(frame, label_text, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"({cx},{cy})", (x1,y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
            pts = list(tr.centers)
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], (255,0,0), 2)
            pred = tr.predict()
            if pred:
                px,py = pred
                cv2.circle(frame, (px,py), 5, (0,255,255), -1)
        return frame

    def _handle_object_alerts(self, scored):
        now = time.monotonic()
        self._forget_old_alert_memory(now)

        #Не поднимаем тревогу по одному и тому же объекту на каждом кадре.
        new_alerts = []
        for tid, tr, threat in scored:
            label = tr.best_label
            center = tr.centroid

            if tid in self.alarmed_track_ids:
                self._remember_alerted_object(tid, label, center, now)
                continue

            if self._is_same_alerted_object(label, center, now):
                self.alarmed_track_ids.add(tid)
                self._remember_alerted_object(tid, label, center, now)
                continue

            self.alarmed_track_ids.add(tid)
            self._remember_alerted_object(tid, label, center, now)
            new_alerts.append((tid, label, center, threat))

        if not new_alerts:
            return

        tid, label, center, threat = new_alerts[0]
        x, y = center
        if len(new_alerts) == 1:
            text = f"Новый объект: ID {tid} | {label} | X {x} Y {y} | угроза {int(threat)}"
        else:
            text = f"Новые объекты: {len(new_alerts)} | первый ID {tid} | {label}"

        self._show_alert_banner(text)
        self._play_alarm_sound()

    def _remember_alerted_object(self, track_id, label, center, timestamp):
        self.alert_memory[track_id] = {
            "label": label,
            "center": center,
            "time": timestamp,
        }

    def _is_same_alerted_object(self, label, center, timestamp):
        for memory in self.alert_memory.values():
            if timestamp - memory["time"] > ALERT_MEMORY_SECONDS:
                continue
            if memory["label"] != label:
                continue
            distance = np.linalg.norm(np.array(memory["center"]) - np.array(center))
            if distance <= ALERT_SAME_OBJECT_DISTANCE:
                return True
        return False

    def _forget_old_alert_memory(self, timestamp):
        for track_id in list(self.alert_memory):
            if timestamp - self.alert_memory[track_id]["time"] > ALERT_MEMORY_SECONDS:
                del self.alert_memory[track_id]

    def _show_alert_banner(self, text):
        if self.alert_reset_job is not None:
            try:
                self.after_cancel(self.alert_reset_job)
            except Exception:
                pass
            self.alert_reset_job = None

        self.alert_label.configure(text=text, fg_color="#991b1b", text_color="#ffffff")
        self.status_label.configure(text=text)
        self.alert_reset_job = self.after(3200, self._reset_alert_banner)

    def _reset_alert_banner(self):
        if not hasattr(self, "alert_label"):
            return
        if self.alert_reset_job is not None:
            self.alert_reset_job = None
        self.alert_label.configure(
            text="Оповещение: новых объектов нет",
            fg_color="#111827",
            text_color="#94a3b8",
        )

    def _play_alarm_sound(self):
        now = time.monotonic()
        if now - self.last_alarm_sound_at < ALARM_SOUND_COOLDOWN_SECONDS:
            return
        self.last_alarm_sound_at = now
        def _play():
            if os.name == 'nt':
                try:
                    import winsound
                    winsound.PlaySound(None, 0)
                    if ALARM_SOUND_PATH.exists():
                        winsound.PlaySound(
                            str(ALARM_SOUND_PATH),
                            winsound.SND_FILENAME | winsound.SND_ASYNC,
                        )
                    else:
                        winsound.MessageBeep(winsound.MB_ICONHAND)
                except Exception:
                    pass
            else:
                self.bell()
        threading.Thread(target=_play, daemon=True).start()

    def _stop_alarm_sound(self):
        if os.name != 'nt':
            return
        try:
            import winsound
            winsound.PlaySound(None, 0)
        except Exception:
            pass

    def _update_table(self, scored):
        if self.table_text is None: return
        self.table_text.delete("0.0","end")
        if not scored:
            self.table_text.insert("0.0", "Нет активных целей\n")
            return
        header = f"{'ID':<4} {'Тип':<12} {'X':<6} {'Y':<6} {'Угроза':<8}\n"
        self.table_text.insert("0.0", header)
        for tid, tr, threat in scored:
            cx, cy = tr.centroid
            line = f"{tid:<4} {tr.best_label:<12} {cx:<6} {cy:<6} {int(threat):<8}\n"
            self.table_text.insert("end", line)

    def _draw_frame(self, frame):
        #Подгоняем кадр под доступное место, но сохраняем пропорции.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        lw = max(self.video_label.winfo_width(), 640)
        lh = max(self.video_label.winfo_height(), 420)
        image.thumbnail((lw, lh), Image.Resampling.LANCZOS)
        self.current_pil_image = image
        self.current_photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self.current_photo, text="")

    def _update_header(self, source=None, mode=None, fps=None, resolution=None):
        if source is not None: self.source_card.configure(text=source)
        if mode is not None: self.mode_card.configure(text=mode)
        if fps is not None: self.fps_card.configure(text=fps)
        if resolution is not None: self.resolution_card.configure(text=resolution)

    def _clear_camera_list(self):
        for w in self.camera_list.winfo_children(): w.destroy()

    def _add_camera_placeholder(self, text):
        lbl = ctk.CTkLabel(self.camera_list, text=text, text_color="#9ca3af", height=52, anchor="w")
        lbl.grid(row=0, column=0, padx=12, pady=12, sticky="ew")

    def close_app(self):
        self.stop_source(clear_screen=False)
        self.destroy()

if __name__ == "__main__":
    app = AirSpaceMonitor()
    app.mainloop()
