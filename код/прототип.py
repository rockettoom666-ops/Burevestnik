from __future__ import annotations

"""Оконный прототип для кейса "Буревестник".

Сейчас это не система распознавания и не трекер. Это первый честный слой:
пользователь выбирает реальную камеру или реальный видеофайл и видит живой
поток. Уже поверх этого экрана потом будет удобно добавлять рамки, координаты
и отслеживание объектов.
"""

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk


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


class BurevestnikPrototype(ctk.CTk):
    #Главное окно приложения.

    #Внутри собраны три вещи: красивый интерфейс, поиск камер и показ кадров.
    


    def __init__(self) -> None:
        super().__init__()

        # Темная тема больше похожа на рабочее место оператора видеонаблюдения
        # и не перетягивает внимание с картинки.
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Буревестник - прототип без отслеживания")
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

        # Флаг защищает от повторного запуска поиска камер по двойному клику.
        self.scanning = False

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
        self.main.grid_rowconfigure(1, weight=1)

        title = ctk.CTkLabel(
            self.sidebar,
            text="Буревестник",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#f8fafc",
        )
        title.grid(row=0, column=0, padx=24, pady=(28, 0), sticky="w")

        subtitle = ctk.CTkLabel(
            self.sidebar,
            text="Прототип просмотра камер\nбез отслеживания объектов",
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
        self.stop_button.grid(row=4, column=0, padx=24, pady=(0, 22), sticky="ew")

        cameras_title = ctk.CTkLabel(
            self.sidebar,
            text="Найденные камеры",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#e5e7eb",
        )
        cameras_title.grid(row=5, column=0, padx=24, pady=(0, 8), sticky="w")

        self.camera_list = ctk.CTkScrollableFrame(
            self.sidebar,
            fg_color="#0f172a",
            corner_radius=12,
            height=270,
        )
        self.camera_list.grid(row=6, column=0, padx=24, pady=(0, 18), sticky="nsew")
        self.camera_list.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(6, weight=1)

        self.status_label = ctk.CTkLabel(
            self.sidebar,
            text="Готов к выбору источника",
            justify="left",
            wraplength=270,
            text_color="#9ca3af",
        )
        self.status_label.grid(row=7, column=0, padx=24, pady=(0, 22), sticky="ew")

        self.header = ctk.CTkFrame(self.main, height=104, fg_color="#0b1018", corner_radius=0)
        self.header.grid(row=0, column=0, sticky="ew", padx=26, pady=(22, 8))
        self.header.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Верхние карточки нужны не для красоты ради красоты: оператор сразу
        # видит источник, режим, FPS и размер кадра.
        self.source_card = self._make_info_block(self.header, "Источник", self.current_source, 0)
        self.mode_card = self._make_info_block(self.header, "Режим", self.current_mode, 1)
        self.fps_card = self._make_info_block(self.header, "FPS", "-", 2)
        self.resolution_card = self._make_info_block(self.header, "Кадр", self.current_resolution, 3)

        # Здесь показывается реальное видео. Позже именно в этом месте появятся
        # рамки объектов и координаты, но сейчас кадр остается чистым.
        self.video_frame = ctk.CTkFrame(self.main, fg_color="#05070b", corner_radius=18)
        self.video_frame.grid(row=1, column=0, sticky="nsew", padx=26, pady=(8, 14))
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

        self.footer = ctk.CTkFrame(self.main, height=58, fg_color="#0b1018", corner_radius=0)
        self.footer.grid(row=2, column=0, sticky="ew", padx=26, pady=(0, 20))
        self.footer.grid_columnconfigure(0, weight=1)

        self.future_label = ctk.CTkLabel(
            self.footer,
            text="Отслеживание объектов сейчас выключено. Этот экран только показывает реальный видеопоток.",
            text_color="#9ca3af",
            anchor="w",
        )
        self.future_label.grid(row=0, column=0, sticky="ew")

    def _make_info_block(
        self,
        parent: ctk.CTkFrame,
        caption: str,
        value: str,
        column: int,
    ) -> ctk.CTkLabel:
        #Делает одну верхнюю карточку с подписью и значением.

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

    def show_empty_view(self) -> None:
        
        self.video_label.configure(
            image=None,
            text="Выбери камеру или загрузи видеофайл",
        )
        self._update_header(source="Источник не выбран", mode="Ожидание", fps="-", resolution="-")

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
        #Показывает найденные камеры в виде кнопок.

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
        #Открывает выбранную камеру и запускает показ живого потока.

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

        self.status_label.configure(text=f"Открыта {camera.title}. На экране идет живой поток с устройства.")
        self._update_header(source=self.current_source, mode=self.current_mode, fps="-", resolution=self.current_resolution)
        self._schedule_frame()

    def open_video_dialog(self) -> None:
        #Открывает системное окно выбора видеофайла.

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
        #Открывает видео для тестового режима.
        

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

        now = time.perf_counter()
        instant_fps = 1.0 / max(now - self.last_frame_at, 1e-6)
        self.last_frame_at = now

        # Сглаживаем FPS, чтобы число в карточке не прыгало каждую миллисекунду.
        self.smoothed_fps = instant_fps if self.smoothed_fps == 0 else self.smoothed_fps * 0.88 + instant_fps * 0.12

        # Главная будущая точка расширения: потом перед _draw_frame(frame)
        # мы добавим детекцию объектов на текущем кадре.
        self._draw_frame(frame)
        self._update_header(fps=f"{self.smoothed_fps:.1f}")
        self._schedule_frame()

    def _draw_frame(self, frame) -> None:
        """Преобразует кадр OpenCV в картинку, которую умеет показывать Tkinter."""

        # OpenCV хранит цвета как BGR, а Pillow/Tkinter ждут RGB.
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


if __name__ == "__main__":
    app = BurevestnikPrototype()
    app.mainloop()
