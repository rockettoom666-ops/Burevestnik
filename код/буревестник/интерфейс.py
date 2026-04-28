from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

from буревестник.детекция import (
    YoloModelChoice,
    detect_objects,
    get_yolo_model_title,
    list_yolo_model_choices,
    load_yolo_model,
)
from буревестник.настройки import (
    MAX_TRACKED_OBJECTS,
    VIDEO_ANALYSIS_FRAME_INTERVAL,
    YOLO_LABELS_RU,
    YOLO_MODEL_ENV_VAR,
)
from буревестник.отрисовка import (
    draw_tracking_overlay,
    draw_waiting_label,
)
from буревестник.сущности import CameraInfo, Track
from буревестник.трекер import SimpleTracker


ALARM_SOUND_PATH = Path(__file__).resolve().parents[2] / "ресурсы" / "тревога.wav"
ALARM_SOUND_COOLDOWN_SECONDS = 2.0


class BurevestnikPrototype(ctk.CTk):
    """Главное окно приложения.

    Внутри остались только вещи, связанные с окном: кнопки, камера, видео,
    таблица и вызов готовых функций детекции/трекинга из отдельных модулей.
    """

    def __init__(self) -> None:
        super().__init__()

        # Темная тема больше похожа на рабочее место оператора видеонаблюдения
        # и не перетягивает внимание с картинки.
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Буревестник - прототип с первым отслеживанием")
        self.geometry("1220x760")
        self.minsize(1060, 680)
        self.configure(fg_color="#0b1018")

        # capture - это текущий источник кадров: камера или выбранное видео.
        # Когда пользователь выбирает другой источник, старый мы освобождаем.
        self.capture: cv2.VideoCapture | None = None

        # frame_job хранит запланированное чтение следующего кадра.
        # Это нужно, чтобы окно не зависало в бесконечном цикле while.
        self.frame_job: str | None = None

        # Tkinter может "забыть" картинку, если не держать ссылку на нее.
        # Поэтому последняя картинка хранится прямо в объекте окна.
        self.current_photo: ImageTk.PhotoImage | None = None

        # Эти поля показываются в верхних карточках интерфейса.
        self.current_source = "Источник не выбран"
        self.current_mode = "Ожидание"
        self.current_resolution = "-"

        # Задержка между кадрами. Для камеры маленькая, для видео берется из FPS.
        self.video_delay_ms = 15
        self.last_frame_at = time.perf_counter()
        self.smoothed_fps = 0.0
        self.frame_counter = 0

        # В отладке можно переключать версии модели без отдельного файла запуска.
        # Список строится по реально найденным .pt-файлам.
        self.model_choices = list_yolo_model_choices()
        self.model_choices_by_label = {choice.label: choice for choice in self.model_choices}
        self.selected_model_choice = self._select_initial_model_choice()
        self.selected_model_path = self.selected_model_choice.path if self.selected_model_choice else None
        if self.selected_model_path is not None:
            os.environ[YOLO_MODEL_ENV_VAR] = str(self.selected_model_path)
        self.video_analysis_frame_interval = self._read_video_analysis_frame_interval()

        # Флаг защищает от повторного запуска поиска камер по двойному клику.
        self.scanning = False

        # Блок детекции/отслеживания. Модель загружается только когда оператор
        # включает переключатель, поэтому обычный просмотр камеры стартует быстро.
        self.tracking_enabled = False
        self.detector_model: Any | None = None
        self.detector_loading = False
        self.tracker = SimpleTracker()
        self.last_tracks: list[Track] = []
        self.alerted_track_ids: set[int] = set()
        self.alert_reset_job: str | None = None
        self.last_alarm_sound_at = 0.0

        # При закрытии окна важно отпустить камеру, иначе она может остаться
        # занятой процессом Python.
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        self._build_layout()
        self.show_empty_view()

    def _build_layout(self) -> None:
        """Собирает весь интерфейс: левую панель, карточки и окно видео."""

        # Макет простой: слева меню, справа большая рабочая зона.
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Левая панель - место, где оператор выбирает, откуда брать поток.
        self.sidebar = ctk.CTkFrame(self, width=330, corner_radius=0, fg_color="#111827")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_columnconfigure(0, weight=1)

        # Правая часть - сама картинка и короткая сводка по источнику.
        self.main = ctk.CTkFrame(self, corner_radius=0, fg_color="#0b1018")
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(2, weight=1)

        title = ctk.CTkLabel(
            self.sidebar,
            text="Буревестник",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#f8fafc",
        )
        title.grid(row=0, column=0, padx=24, pady=(28, 0), sticky="w")

        subtitle = ctk.CTkLabel(
            self.sidebar,
            text="Камеры, видео и первый\nрежим отслеживания объектов",
            justify="left",
            font=ctk.CTkFont(size=14),
            text_color="#9ca3af",
        )
        subtitle.grid(row=1, column=0, padx=24, pady=(4, 24), sticky="w")

        self.camera_button = ctk.CTkButton(
            self.sidebar,
            text="Работать с камерой",
            height=46,
            corner_radius=10,
            fg_color="#0ea5e9",
            hover_color="#0284c7",
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.scan_cameras,
        )
        self.camera_button.grid(row=2, column=0, padx=24, pady=(0, 12), sticky="ew")

        self.video_button = ctk.CTkButton(
            self.sidebar,
            text="Тест на видео",
            height=46,
            corner_radius=10,
            fg_color="#14b8a6",
            hover_color="#0f766e",
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.open_video_dialog,
        )
        self.video_button.grid(row=3, column=0, padx=24, pady=(0, 12), sticky="ew")

        self.stop_button = ctk.CTkButton(
            self.sidebar,
            text="Остановить источник",
            height=40,
            corner_radius=10,
            fg_color="#374151",
            hover_color="#4b5563",
            command=self.stop_source,
        )
        self.stop_button.grid(row=4, column=0, padx=24, pady=(0, 12), sticky="ew")

        self.tracking_switch = ctk.CTkSwitch(
            self.sidebar,
            text="Отслеживание объектов",
            command=self.toggle_tracking,
            onvalue=True,
            offvalue=False,
            progress_color="#14b8a6",
            button_color="#e5e7eb",
            button_hover_color="#ffffff",
            text_color="#e5e7eb",
        )
        self.tracking_switch.grid(row=5, column=0, padx=24, pady=(0, 18), sticky="w")

        self.debug_model_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="#0f172a",
            corner_radius=12,
        )
        self.debug_model_frame.grid(row=6, column=0, padx=24, pady=(0, 14), sticky="ew")
        self.debug_model_frame.grid_columnconfigure(0, weight=1)

        debug_title = ctk.CTkLabel(
            self.debug_model_frame,
            text="Отладка модели",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#e5e7eb",
            anchor="w",
        )
        debug_title.grid(row=0, column=0, padx=12, pady=(10, 4), sticky="ew")

        model_values = [choice.label for choice in self.model_choices] or ["Модели не найдены"]
        self.model_choice_menu = ctk.CTkOptionMenu(
            self.debug_model_frame,
            values=model_values,
            command=self.change_debug_model,
            fg_color="#1f2937",
            button_color="#334155",
            button_hover_color="#475569",
            dropdown_fg_color="#111827",
            dropdown_hover_color="#1f2937",
            text_color="#e5e7eb",
            height=34,
        )
        self.model_choice_menu.grid(row=1, column=0, padx=12, pady=(0, 8), sticky="ew")
        if self.selected_model_choice is not None:
            self.model_choice_menu.set(self.selected_model_choice.label)
        else:
            self.model_choice_menu.configure(state="disabled")

        self.model_debug_label = ctk.CTkLabel(
            self.debug_model_frame,
            text=self._model_debug_text(),
            justify="left",
            wraplength=250,
            text_color="#94a3b8",
            anchor="w",
        )
        self.model_debug_label.grid(row=2, column=0, padx=12, pady=(0, 10), sticky="ew")

        cameras_title = ctk.CTkLabel(
            self.sidebar,
            text="Найденные камеры",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#e5e7eb",
        )
        cameras_title.grid(row=7, column=0, padx=24, pady=(0, 8), sticky="w")

        self.camera_list = ctk.CTkScrollableFrame(
            self.sidebar,
            fg_color="#0f172a",
            corner_radius=12,
            height=145,
        )
        self.camera_list.grid(row=8, column=0, padx=24, pady=(0, 14), sticky="ew")
        self.camera_list.grid_columnconfigure(0, weight=1)

        self.sidebar.grid_rowconfigure(10, weight=1)

        self.status_label = ctk.CTkLabel(
            self.sidebar,
            text="Готов к выбору источника",
            justify="left",
            wraplength=270,
            text_color="#9ca3af",
        )
        self.status_label.grid(row=11, column=0, padx=24, pady=(0, 22), sticky="ew")

        self.header = ctk.CTkFrame(self.main, height=104, fg_color="#0b1018", corner_radius=0)
        self.header.grid(row=0, column=0, sticky="ew", padx=26, pady=(22, 8))
        self.header.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Верхние карточки нужны не для красоты ради красоты: оператор сразу
        # видит источник, режим, FPS и размер кадра.
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
        self.alert_label.grid(row=1, column=0, sticky="ew", padx=26, pady=(0, 6))

        # Здесь показывается реальное видео. Поверх него рисуются рамки,
        # номера и координаты, если включено отслеживание.
        self.video_frame = ctk.CTkFrame(self.main, fg_color="#05070b", corner_radius=18)
        self.video_frame.grid(row=2, column=0, sticky="nsew", padx=26, pady=(8, 14))
        self.video_frame.grid_columnconfigure(0, weight=1)
        self.video_frame.grid_rowconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(
            self.video_frame,
            text="",
            fg_color="#05070b",
            text_color="#94a3b8",
            font=ctk.CTkFont(size=18),
        )
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)

        self.objects_panel = ctk.CTkFrame(self.main, fg_color="#0b1018", corner_radius=0)
        self.objects_panel.grid(row=3, column=0, sticky="ew", padx=26, pady=(0, 10))
        self.objects_panel.grid_columnconfigure(0, weight=1)
        self._build_objects_table(self.objects_panel)

        self.footer = ctk.CTkFrame(self.main, height=58, fg_color="#0b1018", corner_radius=0)
        self.footer.grid(row=4, column=0, sticky="ew", padx=26, pady=(0, 20))
        self.footer.grid_columnconfigure(0, weight=1)

        self.future_label = ctk.CTkLabel(
            self.footer,
            text="Отслеживание пока выключено. Включи переключатель слева, чтобы появились рамки, ID и координаты.",
            text_color="#9ca3af",
            anchor="w",
        )
        self.future_label.grid(row=0, column=0, sticky="ew")

    def _build_objects_table(self, parent: ctk.CTkFrame) -> None:
        """Собирает видимую таблицу объектов под видеопотоком."""

        objects_title_row = ctk.CTkFrame(parent, fg_color="transparent")
        objects_title_row.grid(row=0, column=0, pady=(0, 7), sticky="ew")
        objects_title_row.grid_columnconfigure(0, weight=1)

        objects_title = ctk.CTkLabel(
            objects_title_row,
            text="Объекты на кадре",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#e5e7eb",
            anchor="w",
        )
        objects_title.grid(row=0, column=0, sticky="ew")

        self.objects_count_label = ctk.CTkLabel(
            objects_title_row,
            text="0 сейчас",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#38bdf8",
            anchor="e",
        )
        self.objects_count_label.grid(row=0, column=1, sticky="e")

        self.objects_table = ctk.CTkFrame(
            parent,
            fg_color="#0f172a",
            corner_radius=12,
            height=146,
        )
        self.objects_table.grid(row=1, column=0, sticky="ew")
        self.objects_table.grid_propagate(False)

        for column, width in enumerate((58, 220, 86, 86, 104)):
            self.objects_table.grid_columnconfigure(column, minsize=width, weight=1 if column == 1 else 0)

        # Это настоящая таблица: отдельные колонки проще читать, чем строку
        # текста, особенно когда X/Y быстро меняются на живом видео.
        for column, title_text in enumerate(("ID", "Тип", "X", "Y", "Увер.")):
            header_label = ctk.CTkLabel(
                self.objects_table,
                text=title_text,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#94a3b8",
                anchor="w" if column == 1 else "center",
            )
            header_label.grid(row=0, column=column, padx=(12 if column == 0 else 4, 4), pady=(10, 3), sticky="ew")

        separator = ctk.CTkFrame(self.objects_table, fg_color="#1e293b", height=1)
        separator.grid(row=1, column=0, columnspan=5, padx=12, pady=(0, 4), sticky="ew")

        self.empty_objects_label = ctk.CTkLabel(
            self.objects_table,
            text="Пока объектов нет. Включи отслеживание, и здесь появятся ID, тип и координаты.",
            text_color="#64748b",
            font=ctk.CTkFont(size=13),
        )
        self.empty_objects_label.grid(row=2, column=0, columnspan=5, padx=12, pady=(28, 0), sticky="ew")

        self.object_row_labels: list[list[ctk.CTkLabel]] = []
        for row in range(MAX_TRACKED_OBJECTS):
            labels: list[ctk.CTkLabel] = []
            for column in range(5):
                label = ctk.CTkLabel(
                    self.objects_table,
                    text="",
                    height=20,
                    font=("Consolas", 12),
                    text_color="#e5e7eb",
                    anchor="w" if column == 1 else "center",
                )
                label.grid(row=row + 2, column=column, padx=(12 if column == 0 else 4, 4), pady=1, sticky="ew")
                labels.append(label)
            self.object_row_labels.append(labels)

        self._update_objects_table([])

    def _make_info_block(
        self,
        parent: ctk.CTkFrame,
        caption: str,
        value: str,
        column: int,
    ) -> ctk.CTkLabel:
        # Делает одну верхнюю карточку с подписью и значением.
        block = ctk.CTkFrame(parent, fg_color="#111827", corner_radius=14)
        block.grid(row=0, column=column, padx=6, pady=4, sticky="nsew")
        block.grid_columnconfigure(0, weight=1)

        caption_label = ctk.CTkLabel(
            block,
            text=caption,
            text_color="#8b949e",
            font=ctk.CTkFont(size=12),
            anchor="w",
        )
        caption_label.grid(row=0, column=0, padx=16, pady=(12, 0), sticky="ew")

        value_label = ctk.CTkLabel(
            block,
            text=value,
            text_color="#f8fafc",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w",
        )
        value_label.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="ew")
        return value_label

    def _select_initial_model_choice(self) -> YoloModelChoice | None:
        if not self.model_choices:
            return None
        return self.model_choices[0]

    def _model_debug_text(self) -> str:
        if self.selected_model_choice is None:
            return "Модели не найдены. Положи .pt файл в корень проекта или в папку теста."

        return (
            f"Файл: {self.selected_model_choice.path.name}\n"
            f"Видео: анализ каждый {self.video_analysis_frame_interval}-й кадр"
        )

    def change_debug_model(self, label: str) -> None:
        """Переключает модель прямо из интерфейса отладки."""

        choice = self.model_choices_by_label.get(label)
        if choice is None:
            return

        if self.detector_loading:
            self.status_label.configure(text="Модель уже загружается. Дождись окончания и переключи еще раз.")
            if self.selected_model_choice is not None:
                self.model_choice_menu.set(self.selected_model_choice.label)
            return

        if self.selected_model_choice == choice:
            self.model_debug_label.configure(text=self._model_debug_text())
            return

        self.selected_model_choice = choice
        self.selected_model_path = choice.path
        os.environ[YOLO_MODEL_ENV_VAR] = str(choice.path)
        os.environ["BUREVESTNIK_VIDEO_ANALYSIS_INTERVAL"] = str(choice.video_interval)
        self.video_analysis_frame_interval = self._read_video_analysis_frame_interval()

        self.detector_model = None
        self.tracker.reset()
        self.last_tracks = []
        self.alerted_track_ids.clear()
        self._reset_alert_banner()
        self._stop_object_alert_sound()
        self._update_objects_table([])
        self.model_debug_label.configure(text=self._model_debug_text())

        self.status_label.configure(text=f"Выбрана модель {choice.path.name}.")
        self.future_label.configure(
            text=f"Отладка: активна модель {choice.path.name}. Включи отслеживание или дождись перезагрузки."
        )

        if self.tracking_enabled:
            self.status_label.configure(text=f"Выбрана модель {choice.path.name}. Перезагружаю детектор.")
            self._load_detector_async()

    def _read_video_analysis_frame_interval(self) -> int:
        """Берет шаг анализа видео из окружения или из обычных настроек."""

        raw_value = os.environ.get("BUREVESTNIK_VIDEO_ANALYSIS_INTERVAL", "").strip()
        if not raw_value:
            if self.selected_model_choice is not None:
                return self.selected_model_choice.video_interval
            return VIDEO_ANALYSIS_FRAME_INTERVAL

        try:
            return max(1, int(raw_value))
        except ValueError:
            return VIDEO_ANALYSIS_FRAME_INTERVAL

    def toggle_tracking(self) -> None:
        """Включает или выключает первый режим отслеживания."""

        self.tracking_enabled = bool(self.tracking_switch.get())
        self.tracker.reset()
        self.last_tracks = []
        self.alerted_track_ids.clear()
        self._reset_alert_banner()
        self._stop_object_alert_sound()
        self._update_objects_table([])

        if not self.tracking_enabled:
            self.future_label.configure(
                text="Отслеживание выключено. Видеопоток идет без рамок и номеров."
            )
            self.status_label.configure(text="Отслеживание выключено.")
            return

        self.future_label.configure(
            text=(
                "Отслеживание включено: модель ищет объекты, трекер выдает ID и координаты. "
                f"В тестовом видео анализ идет каждый {self.video_analysis_frame_interval}-й кадр, чтобы поток не тормозил."
            )
        )
        if self.detector_model is None:
            self._load_detector_async()
        else:
            self.status_label.configure(text="Отслеживание включено. Модель уже загружена.")

    def _load_detector_async(self) -> None:
        """Загружает YOLO в отдельном потоке, чтобы окно не подвисало."""

        if self.detector_loading:
            return

        if self.selected_model_path is None:
            self._finish_detector_loading(
                model=None,
                error=FileNotFoundError("Не выбрана модель для отслеживания."),
            )
            return

        self.detector_loading = True
        try:
            model_title = get_yolo_model_title(self.selected_model_path)
        except Exception as error:
            self._finish_detector_loading(model=None, error=error)
            return

        if hasattr(self, "model_choice_menu"):
            self.model_choice_menu.configure(state="disabled")
        self.status_label.configure(text=f"Загружаю модель {model_title}. Первый запуск может занять время.")

        thread = threading.Thread(target=self._load_detector_worker, args=(self.selected_model_path,), daemon=True)
        thread.start()

    def _load_detector_worker(self, model_path: Path) -> None:
        try:
            model = load_yolo_model(model_path)
            self.after(0, lambda: self._finish_detector_loading(model=model, error=None, model_path=model_path))
        except Exception as error:
            self.after(0, lambda: self._finish_detector_loading(model=None, error=error, model_path=model_path))

    def _finish_detector_loading(
        self,
        model: Any | None,
        error: Exception | None,
        model_path: Path | None = None,
    ) -> None:
        self.detector_loading = False
        if hasattr(self, "model_choice_menu") and self.model_choices:
            self.model_choice_menu.configure(state="normal")

        if model_path is not None and self.selected_model_path is not None and model_path != self.selected_model_path:
            return

        if error is not None:
            self.detector_model = None
            self.tracking_enabled = False
            self.tracking_switch.deselect()
            self.future_label.configure(
                text="YOLO не загрузился. Обычный просмотр камеры и видео продолжает работать."
            )
            self.status_label.configure(text=f"Ошибка загрузки YOLO: {error}")
            messagebox.showerror(
                "Не удалось включить отслеживание",
                "YOLO-модель не загрузилась.\n\n"
                f"Ошибка: {error}\n\n"
                "Проверь, что установлена библиотека ultralytics, а файл модели лежит на указанном пути.",
            )
            return

        self.detector_model = model
        model_title = get_yolo_model_title(self.selected_model_path)
        self.status_label.configure(text=f"Модель {model_title} загружена. Отслеживание готово к работе.")
        self.model_debug_label.configure(text=self._model_debug_text())

    def show_empty_view(self) -> None:
        self.video_label.configure(
            image=None,
            text="Выбери камеру или загрузи видеофайл",
        )
        self._update_header(source="Источник не выбран", mode="Ожидание", fps="-", resolution="-")
        self._update_objects_table([])

    def scan_cameras(self) -> None:
        """Ищет подключенные камеры и выводит их список слева."""

        if self.scanning:
            return

        # Перед поиском закрываем активный поток. Так камера не будет занята
        # одновременно просмотром и проверкой.
        self.stop_source(clear_screen=False)
        self.scanning = True
        self.camera_button.configure(state="disabled", text="Ищу камеры...")
        self.status_label.configure(text="Проверяю подключенные камеры. Это может занять несколько секунд.")
        self._clear_camera_list()
        self._add_camera_placeholder("Поиск камер...")

        thread = threading.Thread(target=self._scan_cameras_worker, daemon=True)
        thread.start()

    def _scan_cameras_worker(self) -> None:
        # Поиск камер может занять пару секунд, поэтому он идет в отдельном
        # потоке. Иначе интерфейс завис бы прямо во время сканирования.
        cameras: list[CameraInfo] = []

        # На Windows backend DSHOW обычно быстрее открывает веб-камеры.
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY

        # Проверяем первые 10 индексов. Для учебного прототипа этого хватает:
        # обычные камеры чаще всего имеют номера 0, 1, 2...
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

        # Интерфейс нельзя безопасно менять из фонового потока, поэтому результат
        # передаем обратно в главный поток через after().
        self.after(0, lambda: self._show_camera_results(cameras))

    def _show_camera_results(self, cameras: list[CameraInfo]) -> None:
        # Показывает найденные камеры в виде кнопок.
        self.scanning = False
        self.camera_button.configure(state="normal", text="Работать с камерой")
        self._clear_camera_list()

        if not cameras:
            self._add_camera_placeholder("Камеры не найдены")
            self.status_label.configure(text="Не удалось найти доступные камеры. Проверь подключение или разрешения.")
            return

        self.status_label.configure(text=f"Найдено камер: {len(cameras)}. Выбери нужную для просмотра.")
        for row, camera in enumerate(cameras):
            button = ctk.CTkButton(
                self.camera_list,
                text=f"{camera.title}\n{camera.details}",
                height=62,
                corner_radius=10,
                fg_color="#1f2937",
                hover_color="#334155",
                anchor="w",
                command=lambda item=camera: self.open_camera(item),
            )
            button.grid(row=row, column=0, padx=8, pady=(8, 0), sticky="ew")

    def open_camera(self, camera: CameraInfo) -> None:
        # Открывает выбранную камеру и запускает показ живого потока.
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

        # Для камеры не пытаемся искусственно держать FPS файла: читаем кадры
        # почти сразу, как только окно готово их показать.
        self.video_delay_ms = 15
        self.smoothed_fps = 0.0
        self.last_frame_at = time.perf_counter()
        self.frame_counter = 0

        if self.tracking_enabled:
            self.status_label.configure(text=f"Открыта {camera.title}. Идет живой поток с тестовым отслеживанием.")
        else:
            self.status_label.configure(text=f"Открыта {camera.title}. На экране идет живой поток с устройства.")
        self._update_header(source=self.current_source, mode=self.current_mode, fps="-", resolution=self.current_resolution)
        self._schedule_frame()

    def open_video_dialog(self) -> None:
        # Открывает системное окно выбора видеофайла.
        path = filedialog.askopenfilename(
            title="Выбери видео для теста",
            filetypes=[
                ("Видео", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("Все файлы", "*.*"),
            ],
        )
        if not path:
            return
        self.open_video(Path(path))

    def open_video(self, path: Path) -> None:
        # Открывает видео для тестового режима.
        self.stop_source(clear_screen=False)

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            messagebox.showerror("Видео не открылось", f"Не удалось открыть файл:\n{path}")
            self.show_empty_view()
            return

        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 1 or fps > 120:
            # Иногда файл не сообщает нормальный FPS. Тогда берем спокойные 30,
            # чтобы видео не проигрывалось рывками или слишком быстро.
            fps = 30

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.capture = capture
        self.current_source = path.name
        self.current_mode = "Тест на видео"
        self.current_resolution = f"{width} x {height}" if width and height else "-"

        # Видеофайл показываем с его настоящей скоростью, насколько это возможно.
        self.video_delay_ms = max(1, int(1000 / fps))
        self.smoothed_fps = 0.0
        self.last_frame_at = time.perf_counter()
        self.frame_counter = 0

        if self.tracking_enabled:
            self.status_label.configure(
                text=f"Открыт видеофайл. Для плавности модель анализирует каждый {self.video_analysis_frame_interval}-й кадр."
            )
        else:
            self.status_label.configure(text="Открыт видеофайл. Он проигрывается как реальный поток, без заранее готовых результатов.")
        self._update_header(source=self.current_source, mode=self.current_mode, fps="-", resolution=self.current_resolution)
        self._schedule_frame()

    def stop_source(self, clear_screen: bool = True) -> None:
        """Останавливает текущую камеру или видеофайл."""

        # Сначала отменяем будущий кадр, чтобы после остановки не пришел
        # запоздалый вызов чтения.
        if self.frame_job is not None:
            self.after_cancel(self.frame_job)
            self.frame_job = None

        # Затем освобождаем OpenCV-источник. Это особенно важно для веб-камер.
        if self.capture is not None:
            self.capture.release()
            self.capture = None

        self.tracker.reset()
        self.last_tracks = []
        self.alerted_track_ids.clear()
        self._reset_alert_banner()
        self._stop_object_alert_sound()

        if clear_screen:
            self.status_label.configure(text="Источник остановлен.")
            self.show_empty_view()

    def _schedule_frame(self) -> None:
        # after() ставит чтение следующего кадра в очередь Tkinter.
        # Благодаря этому интерфейс остается живым и кнопки нажимаются.
        self.frame_job = self.after(self.video_delay_ms, self._read_next_frame)

    def _read_next_frame(self) -> None:
        """Берет один кадр из текущего источника и отправляет его на экран."""

        if self.capture is None:
            return

        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.stop_source(clear_screen=False)
            self.video_label.configure(image=None, text="Поток завершен")
            self._update_header(fps="-")
            self.status_label.configure(text="Источник завершился или перестал отдавать кадры.")
            return

        self.frame_counter += 1

        now = time.perf_counter()
        instant_fps = 1.0 / max(now - self.last_frame_at, 1e-6)
        self.last_frame_at = now

        # Сглаживаем FPS, чтобы число в карточке не прыгало каждую миллисекунду.
        self.smoothed_fps = instant_fps if self.smoothed_fps == 0 else self.smoothed_fps * 0.88 + instant_fps * 0.12

        if self.tracking_enabled and self.detector_model is not None:
            frame = self.process_frame_with_tracking(
                frame,
                run_detection=self._should_run_detection_on_frame(),
            )
        elif self.tracking_enabled and self.detector_loading:
            draw_waiting_label(frame, "YOLO загружается...")
        elif self.tracking_enabled and self.detector_model is None:
            draw_waiting_label(frame, "Модель не загружена")
        else:
            self._update_objects_table([])

        self._draw_frame(frame)
        self._update_header(fps=f"{self.smoothed_fps:.1f}")
        self._schedule_frame()

    def _should_run_detection_on_frame(self) -> bool:
        """Для видео не гоняем тяжелую модель на каждом кадре."""

        if self.current_mode != "Тест на видео":
            return True

        return (
            self.frame_counter == 1
            or self.frame_counter % self.video_analysis_frame_interval == 0
        )

    def process_frame_with_tracking(self, frame, run_detection: bool = True):
        """Главная функция отслеживания: детекция, ID, рамки и таблица.

        Эту функцию мы и "привязали" к основному прототипу. Она не запускает
        отдельное окно и не читает видео сама, а получает кадр из нашего GUI.
        Поэтому камера, тестовое видео и будущие режимы остаются в одном месте.
        """

        if not run_detection:
            draw_tracking_overlay(frame, self.last_tracks)
            return frame

        try:
            detections = detect_objects(frame, self.detector_model)
        except Exception as error:
            self.tracking_enabled = False
            self.tracking_switch.deselect()
            self.last_tracks = []
            self.status_label.configure(text=f"Ошибка детекции: {error}")
            self.future_label.configure(text="Отслеживание выключено из-за ошибки модели.")
            return frame

        tracks = self.tracker.update(detections)
        self.last_tracks = tracks
        self._handle_object_alerts(tracks)
        draw_tracking_overlay(frame, tracks)
        self._update_objects_table(tracks)
        return frame

    def _handle_object_alerts(self, tracks: list[Track]) -> None:
        """Подает тревогу только для реально новых ID, а не для уже знакомых объектов."""

        new_tracks = [
            track
            for track in tracks
            if track.track_id not in self.alerted_track_ids
        ]
        if not new_tracks:
            return

        for track in new_tracks:
            self.alerted_track_ids.add(track.track_id)

        first_track = new_tracks[0]
        cx, cy = first_track.center
        label = YOLO_LABELS_RU.get(first_track.label, first_track.label)

        if len(new_tracks) == 1:
            text = f"Новый объект: ID {first_track.track_id} | {label} | X {cx} Y {cy}"
        else:
            text = f"Новые объекты: {len(new_tracks)} | первый ID {first_track.track_id} | {label}"

        self._show_object_alert(text)
        self._play_object_alert_sound()

    def _show_object_alert(self, text: str) -> None:
        if self.alert_reset_job is not None:
            self.after_cancel(self.alert_reset_job)
            self.alert_reset_job = None

        self.alert_label.configure(
            text=text,
            fg_color="#991b1b",
            text_color="#ffffff",
        )
        self.status_label.configure(text=text)
        self.alert_reset_job = self.after(3200, self._reset_alert_banner)

    def _reset_alert_banner(self) -> None:
        if not hasattr(self, "alert_label"):
            return

        if self.alert_reset_job is not None:
            try:
                self.after_cancel(self.alert_reset_job)
            except ValueError:
                pass
            self.alert_reset_job = None

        self.alert_label.configure(
            text="Оповещение: новых объектов нет",
            fg_color="#111827",
            text_color="#94a3b8",
        )

    def _play_object_alert_sound(self) -> None:
        """Воспроизводит WAV-тревогу и перезапускает ее при каждом новом появлении."""

        now = time.monotonic()
        if now - self.last_alarm_sound_at < ALARM_SOUND_COOLDOWN_SECONDS:
            return
        self.last_alarm_sound_at = now

        if os.name != "nt":
            for delay in range(0, 1200, 260):
                self.after(delay, self.bell)
            return

        def worker() -> None:
            try:
                import winsound

                winsound.PlaySound(None, 0)
                if ALARM_SOUND_PATH.exists():
                    winsound.PlaySound(
                        str(ALARM_SOUND_PATH),
                        winsound.SND_FILENAME | winsound.SND_ASYNC,
                    )
                    return

                winsound.MessageBeep(winsound.MB_ICONHAND)
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def _stop_object_alert_sound(self) -> None:
        """Останавливает WAV-тревогу, если оператор выключил источник или закрыл окно."""

        self.last_alarm_sound_at = 0.0

        if os.name != "nt":
            return

        try:
            import winsound

            winsound.PlaySound(None, 0)
        except Exception:
            pass

    def _update_objects_table(self, tracks: list[Track]) -> None:
        """Обновляет живую таблицу объектов под видеопотоком."""

        table = getattr(self, "objects_table", None)
        if table is None:
            return

        visible_tracks = tracks[:MAX_TRACKED_OBJECTS]
        hidden_count = max(len(tracks) - len(visible_tracks), 0)

        if hidden_count:
            self.objects_count_label.configure(text=f"{len(visible_tracks)} из {len(tracks)}")
        else:
            self.objects_count_label.configure(text=f"{len(visible_tracks)} сейчас")

        if visible_tracks:
            self.empty_objects_label.grid_remove()
        else:
            self.empty_objects_label.grid()

        for row, labels in enumerate(self.object_row_labels):
            if row >= len(visible_tracks):
                for label in labels:
                    label.grid_remove()
                    label.configure(text="", fg_color="transparent")
                continue

            track = visible_tracks[row]
            cx, cy = track.center
            label_ru = YOLO_LABELS_RU.get(track.label, track.label)
            confidence = f"{int(track.confidence * 100)}%"
            row_color = "#111c2e" if row % 2 == 0 else "#0f172a"
            values = (
                str(track.track_id),
                label_ru[:12],
                str(cx),
                str(cy),
                confidence,
            )

            for label, value in zip(labels, values):
                label.grid()
                label.configure(text=value, fg_color=row_color)

    def _draw_frame(self, frame) -> None:
        """Преобразует кадр OpenCV в картинку, которую умеет показывать Tkinter."""

        # OpenCV хранит кадры как BGR, а Pillow/Tkinter ждут RGB.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        # Масштабируем изображение под окно, но не ломаем пропорции кадра.
        label_width = max(self.video_label.winfo_width(), 640)
        label_height = max(self.video_label.winfo_height(), 420)
        image.thumbnail((label_width, label_height), Image.Resampling.LANCZOS)

        self.current_photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self.current_photo, text="")

    def _update_header(
        self,
        source: str | None = None,
        mode: str | None = None,
        fps: str | None = None,
        resolution: str | None = None,
    ) -> None:
        # Обновляем только те карточки, для которых пришло новое значение.
        if source is not None:
            self.source_card.configure(text=source)
        if mode is not None:
            self.mode_card.configure(text=mode)
        if fps is not None:
            self.fps_card.configure(text=fps)
        if resolution is not None:
            self.resolution_card.configure(text=resolution)

    def _clear_camera_list(self) -> None:
        # Перед новым поиском убираем старые кнопки камер.
        for widget in self.camera_list.winfo_children():
            widget.destroy()

    def _add_camera_placeholder(self, text: str) -> None:
        # Заглушка внутри списка камер: "поиск", "ничего не найдено" и т.п.
        label = ctk.CTkLabel(
            self.camera_list,
            text=text,
            text_color="#9ca3af",
            height=52,
            anchor="w",
        )
        label.grid(row=0, column=0, padx=12, pady=12, sticky="ew")

    def close_app(self) -> None:
        # Закрываем окно без хвостов: источник отпущен, таймер кадра отменен.
        self.stop_source(clear_screen=False)
        self.destroy()
