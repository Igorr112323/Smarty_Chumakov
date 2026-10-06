"""Извлечение текста из RTF-выгрузки ИПС «Законодательство России».

Зачем: у карточки документа ИПС есть выгрузка ``?savertf=&nd=<N>&page=all`` — это
единственный найденный машинный (не скан) источник полного текста акта. Формат —
RTF в кодировке cp1251: кириллица записана как ``\\'hh``, абзацы — ``\\par``,
таблицы — ``\\cell``/``\\row``, служебные группы открываются ``{\\*``.

Функция ``rtf_to_text`` разбирает этот формат без внешних зависимостей:

* декодирует байты как cp1251 (с запасным utf-8, если встретится);
* выбрасывает служебные группы ``{\\*…}`` и поля ``{\\field…}``;
* превращает ``\\'hh`` в символ, ``\\par``/``\\line`` — в перевод строки,
  ``\\cell``/``\\tab`` — в табуляцию, прочие управляющие слова удаляет;
* убирает экранирование ``\\\\``, ``\\{``, ``\\}``;
* нормализует результат тем же ``normalize_text``, что и весь корпус A3.

Формат разбирается консервативно: всё, что не распознано как управляющее слово,
сохраняется как текст — потерять содержимое акта хуже, чем оставить лишний знак.
"""

from __future__ import annotations

import re
import sys as _sys
from pathlib import Path

# Скрипты запускаются и как модуль (``python -m``), и напрямую (``python scripts/...``),
# поэтому корень модуля добавляется в путь импорта.
_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_MODULE_ROOT))

from scripts.corpus_real import normalize_text  # noqa: E402 - импорт после правки sys.path

# Управляющие слова RTF, которые в тексте акта должны стать переносом строки.
_PARAGRAPH_TAGS = {"par", "line", "pard", "sect", "page"}
# Управляющие слова, которые становятся пробелом/табуляцией (границы ячеек таблиц).
_SPACE_TAGS = {"tab", "cell", "row", "nestcell", "nestrow"}
# Служебные назначения: их содержимое — не текст документа.
_DESTINATIONS = {
    "fonttbl",
    "colortbl",
    "stylesheet",
    "info",
    "pict",
    "object",
    "header",
    "footer",
    "footnote",
    "fldinst",
    "xmlnstbl",
    "listtable",
    "listoverridetable",
    "rsidtbl",
    "generator",
    "themedata",
    "colorschememapping",
    "latentstyles",
    "datastore",
    "filetbl",
    "revtbl",
}

_CONTROL_WORD = re.compile(r"\\([a-zA-Z]+)(-?\d+)?[ ]?")
_HEX_ESCAPE = re.compile(r"\\'(?P<hex>[0-9a-fA-F]{2})")
_GROUP_DESTINATION = re.compile(r"\{\\\*?\\?([a-zA-Z]+)")


def _drop_destination_groups(text: str) -> str:
    """Удалить группы служебных назначений (``{\\*\\fonttbl…}``, ``{\\field…}``).

    Группы в RTF вложены, поэтому счётчик глубины обязателен: иначе вместе с таблицей
    шрифтов можно случайно удалить кусок текста акта.
    """
    result: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        if text[index] == "{" and index + 1 < length and text[index + 1] == "\\":
            rest = text[index + 1 :]
            match = _GROUP_DESTINATION.match("{" + rest[:60])
            name = match.group(1).lower() if match else ""
            if name in _DESTINATIONS or rest.startswith("\\\\*") or name in {"fldinst", "xfld"}:
                depth = 0
                while index < length:
                    if text[index] == "{":
                        depth += 1
                    elif text[index] == "}":
                        depth -= 1
                        if depth == 0:
                            index += 1
                            break
                    index += 1
                continue
        result.append(text[index])
        index += 1
    return "".join(result)


def rtf_to_text(raw: bytes) -> str:
    """Преобразовать байты RTF в обычный текст (cp1251) и нормализовать пробелы."""
    if not raw:
        return ""
    for encoding in ("cp1251", "utf-8"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - обе кодировки не подошли (в реальных выгрузках не встречается)
        text = raw.decode("cp1251", "replace")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _drop_destination_groups(text)
    text = _HEX_ESCAPE.sub(lambda match: bytes([int(match.group("hex"), 16)]).decode("cp1251", "replace"), text)

    def control(match: re.Match[str]) -> str:
        """Заменить управляющее слово на текст (перенос строки, пробел или ничего)."""
        name = match.group(1).lower()
        if name in _PARAGRAPH_TAGS:
            return "\n"
        if name in _SPACE_TAGS:
            return " "
        return ""

    text = _CONTROL_WORD.sub(control, text)
    text = text.replace("\\{", "{").replace("\\}", "}").replace("\\\\", "\\")
    text = text.replace("{", "").replace("}", "")
    return normalize_text(text)
