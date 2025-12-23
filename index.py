import os
import re
import json
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import aiohttp
from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.prompt import Prompt
from rich.align import Align
from rich.status import Status
from rich import box


# ------------------ ENV ------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
GROQ_TIMEOUT_SEC = float(os.getenv("GROQ_TIMEOUT_SEC", "60"))
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "900"))

if not GROQ_API_KEY:
    raise RuntimeError("Не найден GROQ_API_KEY в .env")

console = Console()


# ------------------ DATA ------------------
@dataclass
class Preferences:
    free_text: str = ""
    genres: List[str] = None
    mood: str = ""
    era: str = ""
    language: str = ""
    avoid: List[str] = None
    duration: str = ""
    intensity: str = ""
    want_type: str = ""

    def to_prompt(self) -> str:
        parts: List[str] = []

        if self.free_text:
            parts.append(f"Общие предпочтения: {self.free_text}")

        if self.genres:
            parts.append(f"Жанры: {', '.join(self.genres)}")

        if self.mood:
            parts.append(f"Настроение: {self.mood}")

        if self.era:
            parts.append(f"Эпоха/годы: {self.era}")

        if self.language:
            parts.append(f"Язык: {self.language}")

        if self.duration:
            parts.append(f"Длительность: {self.duration}")

        if self.intensity:
            parts.append(f"Интенсивность: {self.intensity}")

        if self.want_type:
            parts.append(f"Тип: {self.want_type}")

        if self.avoid:
            parts.append(f"Избегать: {', '.join(self.avoid)}")

        return "\n".join(parts).strip()


# ------------------ UI ------------------
BANNER = r"""
   __  ___                 _          ___       _      _
  /  |/  /___ _   _____   (_)___     /   | ____(_)____(_)___  _____
 / /|_/ / __ \ | / / _ \ / / __ \   / /| |/ __/ / ___/ / __ \/ ___/
/ /  / / /_/ / |/ /  __// / / / /  / ___ / /_/ / /__/ / /_/ / /
/_/  /_/\____/|___/\___//_/_/ /_/  /_/  |_\__/_/\___/_/\____/_/
"""

def header_panel() -> Panel:
    txt = Text()
    txt.append(BANNER, style="bold green")
    txt.append("\n")
    txt.append("Movie Advisor - консольный советчик фильмов\n", style="bold white")
    txt.append("Работает через Groq API (LLM)\n", style="green")
    return Panel(Align.center(txt), box=box.DOUBLE, border_style="green")


def info_panel() -> Panel:
    body = Text()
    body.append("Как работает MVP:\n", style="bold white")
    body.append("1) Ты вводишь предпочтения\n", style="green")
    body.append("2) Я уточняю детали\n", style="green")
    body.append("3) Ты получаешь список для просмотра\n", style="green")
    body.append("\nПодсказки:\n", style="bold white")
    body.append("- введи ", style="green")
    body.append("пропуск", style="bold white")
    body.append(" чтобы оставить поле пустым\n", style="green")
    body.append("- введи ", style="green")
    body.append("выход", style="bold white")
    body.append(" чтобы завершить\n", style="green")
    return Panel(body, title="MVP", border_style="green", box=box.ROUNDED)


def ask_field(label: str, default: str = "") -> str:
    raw = Prompt.ask(f"[bold green]{label}[/bold green]", default=default).strip()
    if raw.lower() in {"выход", "exit", "quit", "q"}:
        raise KeyboardInterrupt()
    if raw.lower() in {"пропуск", "skip", "-", "нет", "none"}:
        return ""
    return raw


def choose_from_list(title: str, options: List[str], allow_custom: bool = True, multi: bool = False) -> List[str] | str:
    table = Table(title=title, box=box.SIMPLE, header_style="bold green", border_style="green")
    table.add_column("#", style="green", width=4)
    table.add_column("Вариант", style="bold white")

    for i, opt in enumerate(options, start=1):
        table.add_row(str(i), opt)

    console.print(table)

    hint = "Введи номер" + ("(а) через запятую" if multi else "") + " (или напиши текстом)"
    if not allow_custom:
        hint = "Введи номер" + ("(а) через запятую" if multi else "")

    raw = ask_field(hint, default="").strip()
    if not raw:
        return [] if multi else ""

    raw_low = raw.lower().strip()
    if raw_low in {"пропуск", "skip", "-", "нет", "none"}:
        return [] if multi else ""

    if re.fullmatch(r"[0-9,\s]+", raw):
        nums = [n.strip() for n in raw.split(",") if n.strip().isdigit()]
        picked: List[str] = []
        for n in nums:
            idx = int(n)
            if 1 <= idx <= len(options):
                picked.append(options[idx - 1])
        if multi:
            return list(dict.fromkeys(picked))
        return picked[0] if picked else ""

    if allow_custom:
        if multi:
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            return parts
        return raw

    return [] if multi else ""


def normalize_list(vals: List[str]) -> List[str]:
    out: List[str] = []
    for v in vals:
        vv = (v or "").strip()
        if vv:
            out.append(vv)
    return list(dict.fromkeys(out))


# ------------------ GROQ PROMPTS ------------------
def system_prompt_json() -> str:
    # Тут главное - короткие поля why/plot, иначе модель начинает писать простыни и ломает формат.
    return (
        "Ты - рекомендательная система фильмов и сериалов.\n"
        "Отвечай ТОЛЬКО валидным JSON. Без markdown. Без комментариев.\n"
        "Язык: русский.\n"
        "\n"
        "Формат (JSON):\n"
        "{\n"
        '  \"need_more\": boolean,\n'
        '  \"question\": string,\n'
        '  \"recommendations\": [\n'
        "    {\n"
        '      \"title\": string,\n'
        '      \"year\": string,\n'
        '      \"type\": string,\n'
        '      \"genres\": [string],\n'
        '      \"why\": string,\n'
        '      \"plot\": string\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        "Правила:\n"
        "- Если need_more=true, recommendations пустой массив.\n"
        "- Если need_more=false, question пустая строка.\n"
        "- Верни 10-12 рекомендаций.\n"
        "- why и plot: максимум 140 символов каждое.\n"
        "- Никаких лишних полей.\n"
        "- Не выдумывай очевидный бред.\n"
    )


async def groq_chat(messages: List[Dict[str, str]], force_json: bool = True) -> Tuple[bool, str]:
    """
    force_json=True - пробуем response_format json_object (если поддерживается).
    Если не поддерживается - Groq вернет ошибку, мы отработаем выше.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    payload: Dict[str, Any] = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": GROQ_MAX_TOKENS,
    }

    if force_json:
        payload["response_format"] = {"type": "json_object"}

    timeout = aiohttp.ClientTimeout(total=GROQ_TIMEOUT_SEC)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(url, json=payload, headers=headers) as r:
                body_text = await r.text()
                if r.status != 200:
                    return False, f"HTTP {r.status}: {body_text[:800]}"
                data = json.loads(body_text)
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return True, (content or "").strip()
        except asyncio.TimeoutError:
            return False, "TIMEOUT"
        except Exception as e:
            return False, str(e)


# ------------------ JSON REPAIR / SALVAGE ------------------
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_json_object(raw: str) -> str:
    raw = (raw or "").strip()
    m = JSON_OBJ_RE.search(raw)
    return m.group(0) if m else raw

def strip_json_comments(s: str) -> str:
    s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    return s

def remove_trailing_commas(s: str) -> str:
    s = re.sub(r",\s*([\]}])", r"\1", s)
    return s

def repair_json(raw: str) -> str:
    s = extract_json_object(raw)
    s = strip_json_comments(s)
    s = remove_trailing_commas(s)
    return s.strip()

def try_parse_json(raw: str) -> Tuple[bool, Dict[str, Any], str]:
    candidate = repair_json(raw)
    try:
        obj = json.loads(candidate)
        if not isinstance(obj, dict):
            return False, {}, "JSON не объект"
        obj.setdefault("need_more", False)
        obj.setdefault("question", "")
        obj.setdefault("recommendations", [])
        if not isinstance(obj.get("recommendations"), list):
            obj["recommendations"] = []
        return True, obj, ""
    except Exception as e:
        preview = candidate[:700].replace("\n", "\\n")
        return False, {}, f"Ошибка парсинга JSON: {e}. Пример: {preview}"

def salvage_recommendations(raw: str) -> Tuple[bool, Dict[str, Any], str]:
    """
    Если JSON ломается внутри массива recommendations - пытаемся обрезать до последнего целого объекта.
    Это спасает 80% ситуаций: модель на 8-м фильме запорол кавычку - мы все равно покажем первые 7.
    """
    s = repair_json(raw)

    # Быстро проверяем, может уже норм
    ok, obj, err = try_parse_json(s)
    if ok:
        return True, obj, ""

    # Ищем начало массива recommendations
    rec_key = re.search(r"\"recommendations\"\s*:\s*\[", s)
    if not rec_key:
        return False, {}, err

    start = rec_key.end()  # позиция после '['
    tail = s[start:]

    # Собираем объекты { ... } по балансу скобок
    items: List[str] = []
    depth = 0
    in_str = False
    esc = False
    cur: List[str] = []
    started = False

    for ch in tail:
        if not started:
            if ch == "{":
                started = True
                depth = 1
                cur = ["{"]
            elif ch == "]":
                break
            else:
                continue
            continue

        # started
        cur.append(ch)

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                in_str = False
            continue
        else:
            if ch == "\"":
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    items.append("".join(cur).strip())
                    started = False
                    cur = []

    if not items:
        return False, {}, err

    # Собираем новый JSON с "починенным" массивом
    safe_recs = "[" + ",".join(items) + "]"

    # Заменяем старый recommendations массив на safe_recs грубо
    # 1) вырезаем от "recommendations":[ до закрывающей ] первого уровня
    head = s[:rec_key.start()]
    after_key = s[rec_key.start():]

    # находим закрывающую ] для массива recommendations
    i = after_key.find("[")
    if i == -1:
        return False, {}, err

    idx = i + 1
    depth_arr = 1
    in_str2 = False
    esc2 = False
    while idx < len(after_key):
        ch = after_key[idx]
        if in_str2:
            if esc2:
                esc2 = False
            elif ch == "\\":
                esc2 = True
            elif ch == "\"":
                in_str2 = False
        else:
            if ch == "\"":
                in_str2 = True
            elif ch == "[":
                depth_arr += 1
            elif ch == "]":
                depth_arr -= 1
                if depth_arr == 0:
                    break
        idx += 1

    if depth_arr != 0:
        return False, {}, err

    # after_key[0: i] includes up to '['
    before_arr = after_key[:i]
    after_arr = after_key[idx+1:]  # after closing ']'

    rebuilt = head + before_arr + safe_recs + after_arr
    rebuilt = remove_trailing_commas(rebuilt)

    return try_parse_json(rebuilt)


# ------------------ RENDER ------------------
def render_recommendations(items: List[Dict[str, Any]], title: str = "Список для просмотра") -> None:
    table = Table(
        title=title,
        box=box.HEAVY,
        show_lines=True,
        header_style="bold green",
        border_style="green",
    )

    table.add_column("#", style="bold white", width=3)
    table.add_column("Название", style="bold white", overflow="fold")
    table.add_column("Год", style="green", width=6)
    table.add_column("Тип", style="green", width=8)
    table.add_column("Жанры", style="green", overflow="fold")
    table.add_column("Почему подходит", style="white", overflow="fold")
    table.add_column("Сюжет", style="white", overflow="fold")

    for idx, it in enumerate(items, start=1):
        genres = it.get("genres") or []
        genres_s = ", ".join([str(x) for x in genres[:6]]) if isinstance(genres, list) else str(genres)

        table.add_row(
            str(idx),
            str(it.get("title", "")).strip(),
            str(it.get("year", "")).strip(),
            str(it.get("type", "")).strip(),
            genres_s,
            str(it.get("why", "")).strip(),
            str(it.get("plot", "")).strip(),
        )

    console.print(table)


# ------------------ CORE ------------------
async def get_watchlist(prefs: Preferences) -> Tuple[bool, str, List[Dict[str, Any]]]:
    user_block = prefs.to_prompt() or "Пользователь не указал предпочтения. Задай один короткий уточняющий вопрос."

    messages = [
        {"role": "system", "content": system_prompt_json()},
        {"role": "user", "content": user_block},
    ]

    # 1) Пытаемся через response_format json_object
    ok, raw = await groq_chat(messages, force_json=True)
    if not ok:
        if raw == "TIMEOUT":
            return False, "Таймаут запроса. Попробуй сократить ввод или увеличить GROQ_TIMEOUT_SEC.", []
        # Если response_format не поддержался - пробуем без него
        ok2, raw2 = await groq_chat(messages, force_json=False)
        if not ok2:
            if raw2 == "TIMEOUT":
                return False, "Таймаут запроса. Попробуй снова.", []
            return False, f"Ошибка API: {raw2}", []
        raw = raw2

    # 2) Парсим / ремонтируем
    parsed_ok, obj, err = try_parse_json(raw)
    if not parsed_ok:
        # 3) Salvage: режем recommendations до последнего целого объекта
        salv_ok, salv_obj, salv_err = salvage_recommendations(raw)
        if not salv_ok:
            return False, f"Нейросеть вернула некорректный формат: {salv_err}", []
        obj = salv_obj

    need_more = bool(obj.get("need_more", False))
    question = str(obj.get("question", "") or "").strip()
    recs = obj.get("recommendations") or []
    if not isinstance(recs, list):
        recs = []

    if need_more:
        return True, question or "Какой жанр тебя интересует?", []

    # Если рекомендаций мало (после salvage) - тоже норм, показываем что есть.
    return True, "", recs


# ------------------ OPTIONS ------------------
GENRES = [
    "Триллер", "Психологический триллер", "Детектив", "Криминал", "Нуар", "Саспенс",
    "Хоррор", "Слэшер", "Мистика", "Оккультное", "Паранормальное", "Пост-хоррор",
    "Фантастика", "Научная фантастика", "Космос", "Киберпанк", "Антиутопия", "Постапокалипсис", "Путешествия во времени",
    "Фэнтези", "Темное фэнтези", "Героическое фэнтези", "Сказка",
    "Драма", "Психологическая драма", "Социальная драма", "Семейная драма",
    "Боевик", "Шпионский", "Военное", "Приключения", "Выживание", "Ограбления",
    "Комедия", "Черная комедия", "Сатира", "Романтическая комедия",
    "Историческое", "Биография", "Спорт", "Музыка", "Документальное",
    "Анимация", "Супергероика", "Судебное", "Политический триллер", "Артхаус"
]

MOODS = [
    "Мрачное", "Напряженное", "Тревожное", "Драйвовое", "Спокойное", "Уютное",
    "Вдохновляющее", "Грустное", "Злое", "Саркастичное", "Странное", "Очень умное"
]

ERAS = [
    "70-е", "80-е", "90-е", "2000-2009", "2010-2015", "2016-2020", "2021-2025",
    "Новое (последние 3 года)", "Любое"
]

LANGS = [
    "Любой", "Русский", "Английский", "Корейский", "Японский", "Французский", "Испанский", "Немецкий", "Итальянский"
]

DURATIONS = [
    "<90 минут", "90-120 минут", "2 часа+", "Мини-сериал", "Любая"
]

INTENSITIES = [
    "Спокойный", "Средний", "Жесткий"
]

TYPES = [
    "Фильм", "Сериал", "Любое"
]

AVOID = [
    "Романтика", "Скримеры", "Кровь/жесть", "Депрессивное", "Очень медленно",
    "Пошлость", "Политика", "Слишком сложно", "Подростковое", "Много болтовни"
]


async def main():
    console.clear()
    console.print(header_panel())
    console.print(info_panel())

    while True:
        try:
            console.print()
            console.print(
                Panel(
                    "Опиши, что хочешь посмотреть. Достаточно одной строки.",
                    border_style="green",
                    box=box.ROUNDED,
                )
            )

            free = ask_field("Твои предпочтения", default="что-то умное")

            console.print()
            picked_genres = choose_from_list("Выбор жанров (можно несколько)", GENRES, allow_custom=True, multi=True)
            if isinstance(picked_genres, str):
                picked_genres = [picked_genres]
            picked_genres = normalize_list(picked_genres)

            console.print()
            mood = choose_from_list("Выбор настроения", MOODS, allow_custom=True, multi=False)
            if isinstance(mood, list):
                mood = mood[0] if mood else ""

            console.print()
            era = choose_from_list("Выбор эпохи", ERAS, allow_custom=True, multi=False)
            if isinstance(era, list):
                era = era[0] if era else ""

            console.print()
            lang = choose_from_list("Выбор языка", LANGS, allow_custom=True, multi=False)
            if isinstance(lang, list):
                lang = lang[0] if lang else ""

            console.print()
            duration = choose_from_list("Выбор длительности", DURATIONS, allow_custom=True, multi=False)
            if isinstance(duration, list):
                duration = duration[0] if duration else ""

            console.print()
            intensity = choose_from_list("Выбор интенсивности", INTENSITIES, allow_custom=True, multi=False)
            if isinstance(intensity, list):
                intensity = intensity[0] if intensity else ""

            console.print()
            want_type = choose_from_list("Фильм или сериал", TYPES, allow_custom=True, multi=False)
            if isinstance(want_type, list):
                want_type = want_type[0] if want_type else ""

            console.print()
            avoid_list = choose_from_list("Чего избегать (можно несколько)", AVOID, allow_custom=True, multi=True)
            if isinstance(avoid_list, str):
                avoid_list = [avoid_list]
            avoid_list = normalize_list(avoid_list)

            prefs = Preferences(
                free_text=free,
                genres=picked_genres,
                mood=mood,
                era=era,
                language=lang,
                duration=duration,
                intensity=intensity,
                want_type=want_type,
                avoid=avoid_list,
            )

            with Status("[bold green]Подбираю варианты...[/bold green]", console=console, spinner="dots") as _:
                ok, question, recs = await get_watchlist(prefs)

            if not ok:
                console.print(Panel(str(question), title="Ошибка", border_style="red", box=box.HEAVY))
            else:
                if question:
                    console.print(Panel(question, title="Уточнение", border_style="yellow", box=box.ROUNDED))
                    extra = ask_field("Ответ", default="")
                    prefs.free_text = (prefs.free_text + "\n" + f"Дополнение: {extra}").strip()

                    with Status("[bold green]Подбираю варианты...[/bold green]", console=console, spinner="dots") as _:
                        ok2, question2, recs2 = await get_watchlist(prefs)

                    if not ok2:
                        console.print(Panel(str(question2), title="Ошибка", border_style="red", box=box.HEAVY))
                    else:
                        render_recommendations(recs2, title="Список для просмотра")
                else:
                    render_recommendations(recs, title="Список для просмотра")

            console.print()
            again = Prompt.ask("[bold green]Повторить подбор?[/bold green] (да/нет)", default="да").strip().lower()
            if again not in {"да", "д", "y", "yes"}:
                console.print(Panel("Завершение работы.", border_style="green", box=box.ROUNDED))
                break

        except KeyboardInterrupt:
            console.print()
            console.print(Panel("Завершено пользователем.", border_style="green", box=box.ROUNDED))
            break


if __name__ == "__main__":
    asyncio.run(main())
