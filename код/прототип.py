from __future__ import annotations

"""Точка запуска приложения "Буревестник".

Вся логика теперь разнесена по модулям в папке `буревестник`.
Этот файл специально маленький: его удобно открывать первым и запускать.
"""

from буревестник.интерфейс import BurevestnikPrototype


def main() -> None:
    app = BurevestnikPrototype()
    app.mainloop()


if __name__ == "__main__":
    main()
