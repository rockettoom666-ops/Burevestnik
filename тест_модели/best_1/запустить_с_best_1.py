from __future__ import annotations

import os
import sys
from pathlib import Path


# Этот запуск нужен только для проверки второй модели напарника.
# Основная программа остается той же, меняется только путь к весам YOLO.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "модели" / "best_1.pt"

os.environ["BUREVESTNIK_MODEL"] = str(MODEL_PATH)
os.environ["BUREVESTNIK_VIDEO_ANALYSIS_INTERVAL"] = "15"
sys.path.insert(0, str(PROJECT_ROOT / "код"))

from буревестник.интерфейс import BurevestnikPrototype


def main() -> None:
    app = BurevestnikPrototype()
    app.mainloop()


if __name__ == "__main__":
    main()
