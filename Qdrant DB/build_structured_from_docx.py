# -*- coding: utf-8 -*-
"""
Генерирует structured_*.json из DOCX и сохраняет в processed/

Итоговая версия (на 9.5/10):
- Rooms: учёт «двуспальное место 160*200», fallback-поиск кроватей во всех блоках.
- Hotel: нормализация «Га» → «га».
- FAQ: расширенные автотеги; теги из вопроса и из ответа.
- Contacts: opening_hours (валидированно), whatsapp по тексту и ссылкам, phones_norm (E.164), geo из ссылок Яндекс.Карт.
- Loyalty: корректный захват уровней 1–4 (исправлены паттерны 3–4), срок действия, условия, Telegram-бот.

Зависимости:
    pip install python-docx
"""

import os
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from docx import Document

# ── Пути ─────────────────────────────────────────────────────────────────────
# Для удобства настраиваем пути через переменные окружения, чтобы сценарий
# можно было запускать не только на локальном Windows-компьютере, но и в
# контейнере Amvera или в CI. По умолчанию используем директорию рядом со
# скриптом.
BASE_DIR = Path(os.getenv("HOTEL_DOCS_BASE_DIR", Path(__file__).resolve().parent))
DOCX_DIR = Path(os.getenv("HOTEL_DOCS_SOURCE_DIR", BASE_DIR / "hotel_docs"))
OUT_DIR = Path(os.getenv("HOTEL_DOCS_OUTPUT_DIR", BASE_DIR / "processed"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "rooms":   DOCX_DIR / "Категории номеров и их описание.docx",
    "concept": DOCX_DIR / "Концепция номеров и проживания.docx",
    "contacts":DOCX_DIR / "Наши контакты.docx",
    "hotel":   DOCX_DIR / "Описание отеля и доступных услуг.docx",
    "loyalty": DOCX_DIR / "Программа лояльности.docx",
    "faq":     DOCX_DIR / "Частые вопросы.docx",
}

# ── Утилиты ──────────────────────────────────────────────────────────────────
def fix_typos(text: str) -> str:
    # Сначала устраняем неразрывные пробелы, которые попадали из DOCX и превращались
    # в «невидимые» символы в JSON. Из-за них ранее появлялись странные артефакты
    # (например, удвоенные буквы при склейке слов). Теперь заменяем их на обычные
    # пробелы до остальных преобразований.
    text = text.replace("\xa0", " ")

    # Нормализация Wi-Fi (лат/кирилл i/і, дефис/пробел/ничего)
    text = re.sub(r"\bW[iі][-\s_]*F[iі]\b", "Wi-Fi", text, flags=re.I)
    text = re.sub(r"\bWI[\s_-]*FII\b", "Wi-Fi", text, flags=re.I)
    text = re.sub(r"\bWi-?F\b", "Wi-Fi", text, flags=re.I)

    # Частотные опечатки, появлявшиеся после парсинга
    typo_map = {
        "каализа": "канализа",
        "плотенц": "полотенц",
    }
    for wrong, right in typo_map.items():
        text = re.sub(wrong, right, text, flags=re.I)

    # Пробелы/переводы строк
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def normalize_units(text: str) -> str:
    # «18 Га» → «18 га»
    text = re.sub(r"(\d+)\s*Га\b", r"\1 га", text)
    return text

def docx_to_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")
    doc = Document(str(path))
    parts = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    for tbl in getattr(doc, "tables", []):
        for row in tbl.rows:
            for cell in row.cells:
                t = (cell.text or "").strip()
                if t:
                    parts.append(t)
    return normalize_units(fix_typos("\n".join(parts)))

def gen_keywords(text: str, extra: List[str] = None) -> List[str]:
    kws = set(extra or [])
    tlow = text.lower()
    hints = [
        "терраса","камин","мангал","панорамное остекление","двуспальная кровать","односпальная кровать",
        "детская кроватка","кухня","кондиционер","отопление","wi-fi","сруб","дом","шале","люкс","семейный",
        "сауна","баня","джакузи","панорамные окна","гриль","парковка","вода","тишина"
    ]
    for h in hints:
        if h in tlow:
            kws.add(h)
    for m in re.findall(r"\b\d{2,3}\s*\*\s*\d{2,3}\b", text):  # 160*200 и т.п.
        kws.add(m)
    return sorted(kws)

def normalize_room_name(name: str) -> str:
    if not name:
        return ""
    s = name.lower()
    repl = {"шале комфорт": "шале_комфорт", "вип": "vip"}
    for a, b in repl.items():
        s = s.replace(a, b)
    s = (s
         .replace("ё", "е")
         .replace("й", "и")
         .replace("ь", "")
         .replace("ъ", "")
         .replace(" ", "_"))
    s = re.sub(r"[^a-z0-9_а-я]", "", s)
    return s


def stable_hash(value: str, length: int = 12) -> str:
    """Детерминированный короткий хеш для идентификаторов."""
    value = value or ""
    return hashlib.md5(value.encode("utf-8")).hexdigest()[:length]

# ── Извлечение чисел/флагов ─────────────────────────────────────────────────
NUM_WORDS = {
    "одна":1, "один":1, "одно":1, "по одной":1, "по одному":1,
    "две":2, "два":2, "по две":2, "по два":2,
    "три":3, "четыре":4, "пять":5, "шесть":6, "семь":7, "восемь":8, "девять":9, "десять":10,
}
def _word_to_num(token: str) -> Optional[int]:
    return NUM_WORDS.get(token.lower())

def extract_capacity_max(text_blocks: Dict[str, str]) -> Optional[int]:
    s = " ".join(text_blocks.values())
    m = re.search(r"(?:Проживающих|до)\s*(?:до\s*)?(\d{1,2})\s*(?:человек|гост[еия])", s, flags=re.I)
    return int(m.group(1)) if m else None

def extract_area_sqm(text_blocks: Dict[str, str]) -> Optional[int]:
    s = " ".join(text_blocks.values())
    m = re.search(r"Площадь\s+номера\s*(\d{2,3})\s*кв\.?\s*м", s, flags=re.I)
    return int(m.group(1)) if m else None

def extract_beds(text: str) -> Tuple[int,int,int]:
    """
    (double_beds, single_beds, sofa_beds)
    Понимает: цифры и слова; одиночные упоминания; «спальное/двуспальное место 160*200»; диваны.
    """
    if not text:
        return 0, 0, 0
    src = text

    # Двуспальные/односпальные (цифрами)
    d = sum(int(n) for n in re.findall(r"(\d+)\s*двуспальн\w+", src, flags=re.I))
    s = sum(int(n) for n in re.findall(r"(\d+)\s*односпальн\w+", src, flags=re.I))

    # Двуспальные/односпальные (словами)
    for w, kind in re.findall(r"\b([А-Яа-яёЁ]+)\s+(двуспальн\w+|односпальн\w+)", src, flags=re.I):
        n = _word_to_num(w)
        if not n:
            continue
        if "двуспальн" in kind.lower():
            d += n
        else:
            s += n

    # Одиночные без числа
    if re.search(r"\bдвуспальн\w+\s+кровать\b", src, flags=re.I) and not re.search(r"(\d+|одна|один|одно|две|два|три|четыре|пять)\s+двуспальн", src, flags=re.I):
        d += 1
    if re.search(r"\bодноспальн\w+\s+кровать\b", src, flags=re.I) and not re.search(r"(\d+|одна|один|одно|две|два|три|четыре|пять)\s+односпальн", src, flags=re.I):
        s += 1

    # «спальное место 160*200» или «двуспальное место 160*200»
    if re.search(r"\b(двуспальн\w*\s+место|спальное\s+место)\s*160\*200\b", src, flags=re.I):
        d += 1
    for m in re.findall(r"(\d+)\s*(?:двуспальн\w*\s+мест|спальных?\s*мест[а\w]*)\s*160\*200", src, flags=re.I):
        d += int(m)

    # Диваны
    sofa = 0
    m_sofa = re.search(r"(\d+)\s*раскладн\w*\s*диван", src, flags=re.I)
    if m_sofa:
        sofa += int(m_sofa.group(1))
    elif re.search(r"\bраскладн\w*\s*диван", src, flags=re.I):
        sofa += 1

    return d, s, sofa

def to_bool(text: str, *needles) -> bool:
    tlow = text.lower()
    return any(n in tlow for n in needles)

# ── Контакты: helpers ────────────────────────────────────────────────────────
PHONE_RAW_RE = re.compile(r"(?:\+7|8)\s*[\(\-]?\s*\d{3}\s*[\)\-]?\s*(?:\d[\s\-]?){7}")
DIGITS_RE = re.compile(r"\d+")
URL_RE = re.compile(r"(https?://[^\s]+)", flags=re.I)


def unique_preserve(seq: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def sentence_with_fragment(text: str, fragment: str) -> Optional[str]:
    if not text:
        return None
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sentence in sentences:
        if fragment in sentence:
            return sentence.strip()
    return None

def normalize_phone_e164(phone: str) -> Optional[str]:
    """Приводим российские номера к +7XXXXXXXXXX"""
    if not phone:
        return None
    digits = "".join(DIGITS_RE.findall(phone))
    # 11 цифр с лидирующей '8' или '7'
    if len(digits) == 11 and digits[0] in ("7", "8"):
        return "+7" + digits[1:]
    if len(digits) == 10:  # без кода страны
        return "+7" + digits
    if len(digits) == 11 and phone.strip().startswith("+7"):
        return "+" + digits
    return None

def extract_opening_hours(text: str) -> Optional[str]:
    """
    Валидированное извлечение времени работы:
    - «круглосуточно», «24/7»
    - «9:00–21:00», «10-22», «10.00-22.00»
    - «с 9:00 до 21:00»
    Отбрасывает мусорные совпадения (нечеловеческие часы/минуты).
    """
    tlow = text.lower()
    if "круглосуточно" in tlow or "24/7" in tlow:
        return "24/7"

    def _norm_pair(h1, m1, h2, m2):
        try:
            h1, m1 = int(h1), int(m1 or 0)
            h2, m2 = int(h2), int(m2 or 0)
        except ValueError:
            return None
        if not (0 <= h1 <= 23 and 0 <= m1 <= 59 and 0 <= h2 <= 23 and 0 <= m2 <= 59):
            return None
        return f"{h1:02d}:{m1:02d}-{h2:02d}:{m2:02d}"

    # Форматы "9:00–21:00", "10-22", "10.00-22.00"
    for m in re.finditer(r"(?<!\d)(\d{1,2})[:\.]?(\d{0,2})\s*[-–—]\s*(\d{1,2})[:\.]?(\d{0,2})(?!\d)", text):
        val = _norm_pair(m.group(1), m.group(2) or "00", m.group(3), m.group(4) or "00")
        if val:
            return val

    # «с 9:00 до 21:00»
    m = re.search(r"с\s*(\d{1,2})(?::|\.|h)?(\d{0,2})?\s*до\s*(\d{1,2})(?::|\.|h)?(\d{0,2})?", text, flags=re.I)
    if m:
        val = _norm_pair(m.group(1), m.group(2) or "00", m.group(3), m.group(4) or "00")
        if val:
            return val
    return None

def extract_geo_from_yandex_links(links: List[str]) -> Optional[Dict[str, float]]:
    for url in links:
        if "yandex" not in url.lower():
            continue
        # ищем ll=lon,lat
        m = re.search(r"[?&]ll=([0-9\.\-]+),([0-9\.\-]+)", url)
        if m:
            lon, lat = float(m.group(1)), float(m.group(2))
            return {"lat": lat, "lon": lon}
        # иногда встречается 'map=lat,lon'
        m = re.search(r"[?&]map=([0-9\.\-]+),([0-9\.\-]+)", url)
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
            return {"lat": lat, "lon": lon}
    return None

# ── Категории ────────────────────────────────────────────────────────────────
def build_rooms(text: str) -> List[Dict]:
    entries = []
    parts = re.split(r"\n(?=Номер категории\s+)", text)
    patterns = {
        "Возможные варианты размещения": r"^Возможные варианты размещения\s*:\s*(.*)$",
        "Тип строения": r"^Тип строения\s*[:\-]\s*(.*)$",
        "Характеристики помещения": r"^Характеристики помещения\s*:\s*(.*)$",
        "Спальные места": r"^Спальные места(?:\s*в\s*номере)?\s*:\s*(.*)$",
        "Оснащение": r"^(?:В номер есть|В номере есть)\s*:\s*(.*)$",
        "Дополнительно": r"^Для гостей номера доступны?а?\s*(.*)$",
    }

    for part in parts:
        part = part.strip()
        if not part.startswith("Номер категории"):
            continue
        lines = [fix_typos(l.strip()) for l in part.splitlines() if l.strip()]
        title = lines[0]
        subcat = title.replace("Номер категории", "").strip()

        text_blocks = {
            "Описание": "",
            "Возможные варианты размещения": "",
            "Тип строения": "",
            "Характеристики помещения": "",
            "Спальные места": "",
            "Оснащение": "",
            "Дополнительно": "",
        }

        current_key = None
        for line in lines[1:]:
            matched_key = None
            for key, pat in patterns.items():
                m = re.match(pat, line, flags=re.I)
                if m:
                    matched_key = key
                    value = m.group(1).strip()
                    text_blocks[key] = (text_blocks[key] + " " + value).strip()
                    break
            if not matched_key:
                if current_key:
                    text_blocks[current_key] = (text_blocks[current_key] + " " + line).strip()
                else:
                    text_blocks["Описание"] = (text_blocks["Описание"] + " " + line).strip()
            else:
                current_key = matched_key

        # Переброс лишнего из «Спальные места» в «Оснащение»
        sm = text_blocks.get("Спальные места", "")
        if sm and re.search(r"\b(wi-?fi|кондиционер|теплые полы|чайник|посуд|телевизор|холодильник|фен|полотенц|кофемашин|печь|микроволнов)\b", sm, flags=re.I):
            beds, sep, tail = sm.partition(".")
            if sep:
                text_blocks["Спальные места"] = beds.strip().rstrip(";,. ")
                text_blocks["Оснащение"] = (text_blocks.get("Оснащение", "") + " " + tail).strip()
            else:
                text_blocks["Оснащение"] = (text_blocks.get("Оснащение", "") + " " + sm).strip()
                text_blocks["Спальные места"] = re.sub(r";?\s*(Wi-?Fi.*)$", "", text_blocks["Спальные места"], flags=re.I).strip()

        # Пост-очистка
        text_blocks = {k: fix_typos(v).lstrip(": ,;").strip() for k, v in text_blocks.items() if v}
        if "Оснащение" in text_blocks:
            text_blocks["Оснащение"] = re.sub(r"^[\s,:;]+", "", text_blocks["Оснащение"]).strip()
        if "Спальные места" in text_blocks:
            text_blocks["Спальные места"] = re.sub(r"[\s,;]+$", "", text_blocks["Спальные места"]).strip()

        # Числа/флаги
        capacity_max = extract_capacity_max(text_blocks)
        area_sqm = extract_area_sqm(text_blocks)

        # Кровати: при пустоте/скудности блока ищем во всех
        bed_text_primary = text_blocks.get("Спальные места", "")
        bed_text_fallback = " ".join(text_blocks.values())
        use_all = (
            not bed_text_primary or
            not re.search(r"(двуспальн|односпальн|раскладн\w*\s*диван|спальное\s+место\s*160\*200|двуспальн\w*\s+место\s*160\*200)", bed_text_primary, flags=re.I)
        )
        bed_src = bed_text_fallback if use_all else bed_text_primary
        db, sb, sof = extract_beds(bed_src)

        all_text = " ".join(text_blocks.values())
        features = {
            "has_fireplace": to_bool(all_text, "камин", "каминный зал", "русская печь"),
            "has_kitchen": to_bool(all_text, "на кухне", "кухне ", "кофемашина", "мойка, стол", "комплект посуды"),
            "has_terrace": to_bool(all_text, "терраса"),
            "has_bbq": to_bool(all_text, "мангал", "мангальная"),
            "has_ac": to_bool(all_text, "кондиционер"),
            "has_heating": to_bool(all_text, "отопление", "теплые полы"),
            "has_wifi": to_bool(all_text, "wi-fi"),
            "panoramic_windows": to_bool(all_text, "панорамное остекление"),
            "is_log_house": to_bool(all_text, "сруб"),
        }
        numbers = {
            "capacity_max": capacity_max,
            "area_sqm": area_sqm,
            "beds_double": db or 0,
            "beds_single": sb or 0,
            "sofa_beds": sof or 0,
        }

        room_slug = normalize_room_name(subcat)
        if not room_slug:
            room_slug = f"auto_{stable_hash(title)}"

        entries.append({
            "id": f"rooms:{room_slug}",
            "category": "rooms",
            "subcategory": subcat,
            "title": title,
            "text_blocks": text_blocks,
            "numbers": numbers,
            "features": features,
            "keywords": gen_keywords(f"{title}. " + all_text, extra=[subcat.lower(), "номер"]),
            "room_name_norm": normalize_room_name(subcat),
            "source": "Категории номеров и их описание"
        })

    return entries

def build_concept(text: str) -> List[Dict]:
    headings = [
        "Приватность", "Экологичность", "Комфорт", "Оригинальные Интерьеры",
        "Отдых на природе", "Разнообразие вариантов", "Варианты номеров"
    ]
    h_pat = "|".join(map(re.escape, headings))
    entries: List[Dict] = []

    for g in re.finditer(rf"(?P<h>{h_pat})\s*(?P<body>.*?)(?=(?:{h_pat})\s*|\Z)", text, flags=re.S):
        h = g.group("h").strip()
        body = re.sub(r"\s+", " ", g.group("body")).strip()
        body = re.sub(r"\b(Шале)\s+(Гранд\s+Шале)\b", r"\1, \2", body)  # косметика
        if not body:
            continue
        tag_map = {
            "Приватность": "privacy",
            "Экологичность": "eco",
            "Комфорт": "comfort",
            "Оригинальные Интерьеры": "design",
            "Отдых на природе": "nature",
            "Разнообразие вариантов": "diversity",
            "Варианты номеров": "inventory",
        }
        tag = tag_map.get(h, h.lower())
        entries.append({
            "id": f"concept:{tag}",
            "category": "concept",
            "tag": tag,
            "title": h,
            "text": body,
            "keywords": gen_keywords(body, extra=[h.lower()]),
            "source": "Концепция номеров и проживания"
        })

    # Склейка повторов/хвостов
    merged: List[Dict] = []
    for item in entries:
        if merged and item["title"] == merged[-1]["title"]:
            merged[-1]["text"] = (merged[-1]["text"].rstrip(", ") + " " + item["text"].lstrip(", ")).strip()
            merged[-1]["keywords"] = sorted(set(merged[-1].get("keywords", []) + item.get("keywords", [])))
        elif item["text"].lstrip().startswith(",") and merged:
            merged[-1]["text"] = (merged[-1]["text"].rstrip(", ") + " " + item["text"].lstrip(", ")).strip()
        else:
            merged.append(item)
    for it in merged:
        it["text"] = re.sub(r"\b(Шале)(,\s*\1\b)+", r"\1", it["text"])
    return merged

def build_contacts(text: str) -> List[Dict]:
    entries: List[Dict] = []
    phone_pat = PHONE_RAW_RE

    booking = re.search(r"(брони\w*|заказ\w*|онлайн\s*бронир\w*).{0,120}(" + phone_pat.pattern + r")", text, flags=re.I | re.S)
    reception = re.search(r"(ресепшен|администратор|стойка).{0,100}(" + phone_pat.pattern + r")", text, flags=re.I)
    restaurant = re.search(r"(ресторан|кафе|бар).{0,100}(" + phone_pat.pattern + r")", text, flags=re.I)

    all_links = unique_preserve(URL_RE.findall(text))
    yandex_links = [l for l in all_links if "yandex" in l.lower()]
    general_hours = extract_opening_hours(text)

    def window_around(match: re.Match, radius: int = 200) -> str:
        start, end = match.span(2)
        return text[max(0, start - radius): min(len(text), end + radius)]

    def links_from_context(ctx: str) -> List[str]:
        return unique_preserve(URL_RE.findall(ctx))

    def detect_whatsapp_context(focus_text: str, links_ctx: List[str], raw_phone: str) -> bool:
        if any("wa.me" in l.lower() or "whatsapp" in l.lower() for l in links_ctx):
            return True
        sanitized = "".join(DIGITS_RE.findall(raw_phone))
        if not sanitized:
            return False
        for match in re.finditer(r"whatsapp|ватсап", focus_text, flags=re.I):
            start = max(0, match.start() - 40)
            end = min(len(focus_text), match.end() + 40)
            snippet = focus_text[start:end]
            digits_near = "".join(DIGITS_RE.findall(snippet))
            if sanitized in digits_near:
                return True
        return False

    def normalize_context(ctx: str) -> str:
        return re.sub(r"\s+", " ", ctx).strip()

    def pack_contact(contact_id: str, ctype: str, title: str, phone_match: Optional[re.Match], keywords_extra: Optional[List[str]] = None):
        if not phone_match:
            return
        raw = phone_match.group(2)
        ctx = window_around(phone_match)
        ctx_clean = normalize_context(ctx)
        local_links = links_from_context(ctx)
        hours_local = extract_opening_hours(ctx)
        hours = hours_local or (general_hours if ctype in {"booking", "reception"} else None)
        sentence = sentence_with_fragment(ctx_clean, raw)
        start, end = phone_match.span(2)
        focus_raw = text[max(0, start - 80): min(len(text), end + 80)]
        focus_clean = normalize_context(focus_raw)
        has_wa = detect_whatsapp_context(focus_clean, local_links, raw)
        if not has_wa and sentence:
            has_wa = bool(re.search(r"whatsapp|ватсап", sentence, flags=re.I))
        phones_norm = list(filter(None, [normalize_phone_e164(raw)]))
        base_text = sentence or ctx_clean or f"{title}: {raw}"
        if hours and hours not in base_text:
            suffix = "" if base_text.endswith(".") else "."
            base_text = f"{base_text}{suffix} Часы: {hours}"
        keywords = ["контакты", ctype, "телефон"]
        if has_wa:
            keywords.append("whatsapp")
        if keywords_extra:
            keywords.extend(keywords_extra)
        entries.append({
            "id": contact_id,
            "category": "contacts",
            "contact_type": ctype,
            "title": title,
            "phone": raw,
            "phones": [raw],
            "phones_norm": phones_norm,
            "hours": hours,
            "opening_hours": hours,
            "whatsapp": has_wa,
            "links": local_links,
            "geo": None,
            "text": base_text,
            "keywords": sorted(set(keywords)),
            "source": "Наши контакты"
        })

    pack_contact("contacts:booking", "booking", "Контакты для бронирования", booking, keywords_extra=["бронирование"])
    pack_contact("contacts:reception", "reception", "Телефон ресепшена", reception, keywords_extra=["ресепшен"])
    pack_contact("contacts:restaurant", "restaurant", "Телефон ресторана", restaurant, keywords_extra=["ресторан"])

    social_links = [l for l in all_links if any(x in l.lower() for x in ("instagram.com", "t.me", "vk.com", "youtube.com"))]
    if social_links:
        entries.append({
            "id": "contacts:social",
            "category": "contacts",
            "contact_type": "social",
            "title": "Социальные сети",
            "links": social_links,
            "text": " ; ".join(social_links),
            "keywords": sorted({"соцсети"} | ({"instagram"} if any("instagram" in l.lower() for l in social_links) else set()) | ({"telegram"} if any("t.me" in l.lower() for l in social_links) else set()) | ({"vk"} if any("vk.com" in l.lower() for l in social_links) else set())),
            "source": "Наши контакты"
        })

    m_dir = re.search(r"Как добраться на машине\s*(.+)$", text, flags=re.S)
    if m_dir:
        entries.append({
            "id": "contacts:directions_car",
            "category": "contacts",
            "contact_type": "directions",
            "title": "Как добраться на машине",
            "links": yandex_links,
            "geo": extract_geo_from_yandex_links(yandex_links),
            "text": re.sub(r"\s+", " ", m_dir.group(1)).strip(),
            "keywords": ["как добраться", "машина", "маршрут", "навигатор", "яндекс"],
            "source": "Наши контакты"
        })

    return entries

HOTEL_SECTION_META = {
    "about": {
        "subcategory": "Общее описание",
        "title": "Общее описание отеля",
        "keywords": {"эко-отель", "отель", "загородный", "отдых"},
    },
    "audience": {
        "subcategory": "Кому подходит отдых",
        "title": "Кому подходит отдых",
        "keywords": {"семьи", "пары", "друзья", "тимбилдинг"},
    },
    "location": {
        "subcategory": "Расположение",
        "title": "Расположение",
        "keywords": {"расположение", "локация", "Минское шоссе", "Можайский район", "деревня Власово", "100 км"},
    },
    "territory": {
        "subcategory": "Территория",
        "title": "Территория",
        "keywords": {"территория", "га", "тишина", "огороженная"},
    },
    "services": {
        "subcategory": "Услуги и инфраструктура",
        "title": "Услуги и инфраструктура",
        "keywords": {"услуги", "активности", "инфраструктура", "развлечения"},
        "use_heading_title": True,
    },
    "dining": {
        "subcategory": "Питание и рестораны",
        "title": "Питание и рестораны",
        "keywords": {"ресторан", "питание", "кафе", "бар", "завтрак"},
        "use_heading_title": True,
    },
    "wellness": {
        "subcategory": "SPA и бани",
        "title": "SPA и бани",
        "keywords": {"баня", "сауна", "spa", "wellness"},
        "use_heading_title": True,
    },
    "kids": {
        "subcategory": "Для детей",
        "title": "Инфраструктура для детей",
        "keywords": {"дети", "семейный", "игровая"},
        "use_heading_title": True,
    },
    "events": {
        "subcategory": "Мероприятия и события",
        "title": "Мероприятия и события",
        "keywords": {"мероприятия", "свадьба", "банкет", "тимбилдинг"},
        "use_heading_title": True,
    },
    "nature": {
        "subcategory": "Отдых на природе",
        "title": "Отдых на природе",
        "keywords": {"природа", "лес", "панорама", "тишина"},
        "use_heading_title": True,
    },
}

HOTEL_SECTION_ORDER = [
    "about",
    "audience",
    "location",
    "territory",
    "services",
    "dining",
    "wellness",
    "kids",
    "events",
    "nature",
]


def is_heading_candidate(line: str) -> bool:
    if not line:
        return False
    stripped = line.strip()
    if len(stripped) > 80:
        return False
    if stripped.endswith(":"):
        return True
    if any(ch in stripped for ch in ".!?;:"):
        return False
    words = stripped.split()
    if not words:
        return False
    return all(word.isupper() or (len(word) > 1 and word[0].isupper() and word[1:].islower()) for word in words)


def classify_hotel_paragraph(text: str, heading: Optional[str]) -> List[str]:
    heading_lower = (heading or "").lower()
    lower = text.lower()
    combined = f"{heading_lower} {lower}".strip()

    if "подойдет" in lower or "кому подходит" in heading_lower:
        return ["audience"]

    if any(token in combined for token in ("располож", "локац", "адрес", "находит", "доехать", "маршрут", "км от", "дорог")):
        return ["location"]

    keys: List[str] = []

    if any(token in combined for token in ("территор", "га", "участок", "гектар")):
        keys.append("territory")

    if any(token in combined for token in ("услуг", "инфраструктур", "активн", "развлеч", "сервис", "спорт", "прокат", "аренд", "оборуд", "конференц", "катан", "каток", "экскурс", "бассейн")):
        keys.append("services")

    if any(token in combined for token in ("питан", "ресторан", "кафе", "бар", "меню", "завтрак", "кухн", "гриль", "банкета")):
        keys.append("dining")

    if any(token in combined for token in ("баня", "сауна", "спа", "spa", "хамам", "джакузи", "массаж", "wellness", "купель")):
        keys.append("wellness")

    if any(token in combined for token in ("дет", "аниматор", "игров", "площадк", "семейн", "подрост")):
        keys.append("kids")

    if any(token in combined for token in ("меропр", "свадь", "банкет", "тимбилдинг", "корпоратив", "ивент", "конференц")):
        keys.append("events")

    if any(token in combined for token in ("природ", "лес", "озеро", "тишин", "воздух", "прогулк", "панорам")):
        keys.append("nature")

    if not keys:
        return ["about"]
    return keys


def build_hotel(text: str) -> List[Dict]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    sections: List[Tuple[Optional[str], List[str]]] = []
    current_heading: Optional[str] = None
    current_body: List[str] = []

    for line in lines:
        if is_heading_candidate(line):
            if current_body:
                sections.append((current_heading, current_body))
                current_body = []
            current_heading = line.rstrip(":").strip()
        else:
            current_body.append(line)

    if current_body:
        sections.append((current_heading, current_body))

    entries_map: Dict[str, Dict] = {}

    for heading, body_parts in sections:
        for paragraph in body_parts:
            keys = classify_hotel_paragraph(paragraph, heading)
            for key in keys:
                meta = HOTEL_SECTION_META.get(key, {
                    "subcategory": key.title(),
                    "title": key.title(),
                    "keywords": set(),
                })
                entry = entries_map.setdefault(key, {
                    "subcategory": meta["subcategory"],
                    "title": meta["title"],
                    "title_override": False,
                    "parts": [],
                    "keywords": set(meta.get("keywords", set())),
                })
                if heading and meta.get("use_heading_title") and not entry["title_override"]:
                    entry["title"] = heading
                    entry["title_override"] = True
                if paragraph not in entry["parts"]:
                    entry["parts"].append(paragraph)
                entry["keywords"].update(gen_keywords(paragraph))

    results: List[Dict] = []
    seen_keys = set()

    for key in HOTEL_SECTION_ORDER:
        entry = entries_map.get(key)
        if not entry:
            continue
        text_body = " ".join(entry["parts"]).strip()
        if not text_body:
            continue
        meta = HOTEL_SECTION_META.get(key, {})
        keywords_base = set(entry.get("keywords", set()))
        keywords = sorted(set(gen_keywords(text_body, extra=list(meta.get("keywords", [])))) | keywords_base)
        results.append({
            "id": f"hotel:{key}",
            "category": "hotel",
            "subcategory": entry["subcategory"],
            "title": entry["title"],
            "text": text_body,
            "keywords": keywords,
            "source": "Описание отеля и доступных услуг",
        })
        seen_keys.add(key)

    for key, entry in entries_map.items():
        if key in seen_keys:
            continue
        text_body = " ".join(entry["parts"]).strip()
        if not text_body:
            continue
        meta = HOTEL_SECTION_META.get(key, {})
        keywords_base = set(entry.get("keywords", set()))
        keywords = sorted(set(gen_keywords(text_body, extra=list(meta.get("keywords", [])))) | keywords_base)
        results.append({
            "id": f"hotel:{key}",
            "category": "hotel",
            "subcategory": entry["subcategory"],
            "title": entry["title"],
            "text": text_body,
            "keywords": keywords,
            "source": "Описание отеля и доступных услуг",
        })

    return results

def build_loyalty(text: str) -> List[Dict]:
    entries = [{
        "id": "loyalty:overview",
        "category": "loyalty",
        "subcategory": "Общие условия",
        "title": "Участие и базовые условия",
        "text": "Каждый гость становится участником программы лояльности. Для всех участников — бесплатные 2 часа раннего заезда (при наличии возможности).",
        "keywords": ["программа лояльности", "ранний заезд"],
        "source": "Программа лояльности"
    }]

    levels = [
        (1, r"Уровень лояльности\s*1\s*СЕЗОН[А]?\s*после 1-?го приезда:?\s*(.+?)(?=Уровень лояльности|Срок действия|Начисления|Чтобы проверить|\Z)"),
        (2, r"Уровень лояльности\s*2\s*СЕЗОНА\s*после 2-?го приезда:?\s*(.+?)(?=Уровень лояльности|Срок действия|Начисления|Чтобы проверить|\Z)"),
        (3, r"Уровень лояльности\s*3\s*СЕЗОНА\s*после 3-?го приезда:?\s*(.+?)(?=Уровень лояльности|Срок действия|Начисления|Чтобы проверить|\Z)"),
        (4, r"Уровень лояльности\s*4\s*СЕЗОНА\s*после 4-?го приезда:?\s*(.+?)(?=Срок действия|Начисления|Чтобы проверить|\Z)"),
    ]
    for lvl, pat in levels:
        m = re.search(pat, text, flags=re.S | re.I)
        if m:
            txt = re.sub(r"\s+", " ", m.group(1)).strip()
            entries.append({
                "id": f"loyalty:season_{lvl}",
                "category": "loyalty",
                "subcategory": f"{lvl} СЕЗОН(А)" if lvl > 1 else "1 СЕЗОН",
                "title": f"Уровень лояльности {lvl} СЕЗОНА" if lvl > 1 else "Уровень лояльности 1 СЕЗОН",
                "text": txt,
                "keywords": ["бонусы", "привилегии", f"{lvl} сезон"],
                "source": "Программа лояльности"
            })

    m_expiry = re.search(r"Срок действия бонусов\s*(\d+\s*месяц[аев]*)", text, flags=re.I)
    if m_expiry:
        entries.append({
            "id": "loyalty:expiry",
            "category": "loyalty",
            "subcategory": "Срок действия бонусов",
            "title": "Срок действия бонусов",
            "text": f"Срок действия бонусов {m_expiry.group(1)} с даты начисления.",
            "keywords": ["срок действия", "бонусы"],
            "source": "Программа лояльности"
        })

    m_rule = re.search(r"Начисления.*?только по бронированиям.*?(usadba4\.ru)", text, flags=re.I | re.S)
    if m_rule:
        entries.append({
            "id": "loyalty:eligibility",
            "category": "loyalty",
            "subcategory": "Условия начисления",
            "title": "Условия начисления бонусов",
            "text": "Начисления по программе лояльности производятся только по бронированиям, сделанным через сайт usadba4.ru",
            "keywords": ["начисления", "условия", "сайт"],
            "source": "Программа лояльности"
        })

    m_bot = re.search(r"(https?://t\.me/[^\s]+)", text, flags=re.I)
    if m_bot:
        entries.append({
            "id": "loyalty:telegram_bot",
            "category": "loyalty",
            "subcategory": "Проверка бонусов",
            "title": "Проверка бонусов в Telegram-боте",
            "text": f"Проверить бонусы: {m_bot.group(1)} (вход по номеру телефона, указанному при выезде).",
            "keywords": ["telegram", "бот", "бонусы"],
            "source": "Программа лояльности"
        })

    return entries

# ── FAQ ─────────────────────────────────────────────────────────────────────
FAQ_KEYS = [
    "мангал","завтрак","ресторан","питани","заезд","выезд","wi-fi","интернет",
    "дет","живот","заряд","электроавтомоб","планировк","территори","отмена","перенос","оплат",
    "включен","экскурс","гостев","вода","коммуникац","тишин","парковк","баня","сауна","камин","террас"
]
def faq_topic(s: str):
    s = s.lower()
    for k in FAQ_KEYS:
        if k in s:
            return k
    return None

def tags_from_text(t: str) -> List[str]:
    tlow = t.lower()
    tags = []
    if "wi-fi" in tlow or "интернет" in tlow: tags.append("интернет")
    if "питани" in tlow or "ресторан" in tlow or "завтрак" in tlow: tags.append("питание")
    if "включен" in tlow: tags.append("включено/стоимость")
    if "дет" in tlow: tags.append("дети")
    if "живот" in tlow or "соба" in tlow or "кошк" in tlow: tags.append("животные")
    if "бронир" in tlow or "оплат" in tlow or "отмена" in tlow or "перенос" in tlow: tags.append("бронирование")
    if "мангал" in tlow or "шашлык" in tlow: tags.append("мангал")
    if "заезд" in tlow or "выезд" in tlow: tags.append("заезд/выезд")
    if "территори" in tlow or "планировк" in tlow: tags.append("территория")
    if "заряд" in tlow or "электроавтомоб" in tlow: tags.append("электромобиль")
    if "экскурс" in tlow or "гостев" in tlow: tags += ["посещения","территория"]
    if "вода" in tlow or "коммуникац" in tlow: tags.append("коммуникации")
    if "тишин" in tlow: tags.append("тишина")
    if "парковк" in tlow: tags.append("парковка")
    if "баня" in tlow or "сауна" in tlow: tags.append("баня/сауна")
    if "камин" in tlow: tags.append("камин")
    if "террас" in tlow: tags.append("терраса")
    return tags

def build_faq(text: str) -> List[Dict]:
    entries = []
    qa_raw = re.findall(r"Вопрос:\s*(.+?)\s*Ответ:\s*(.+?)(?=Вопрос:|\Z)", text, flags=re.S)
    for q_raw, a in qa_raw:
        sub_qs = re.split(r"\s*Вопрос:\s*", q_raw)
        sub_qs = [q.strip(" .") for q in sub_qs if q.strip()]
        qs = sub_qs if len(sub_qs) > 1 else [q_raw.strip(" .")]

        a_clean = fix_typos(re.sub(r"\s+", " ", a).strip(" ."))
        a_topic = faq_topic(a_clean)

        for q in qs:
            q_clean = fix_typos(re.sub(r"\s+", " ", q).strip(" ."))
            q_topic = faq_topic(q_clean)
            if q_topic and (a_topic is None or q_topic not in a_clean.lower()):
                continue  # жесткий конфликт — пропускаем

            # Авто-теги из ВОПРОСА и ИЗ ОТВЕТА
            tags = tags_from_text(q_clean) + tags_from_text(a_clean)

            entries.append({
                "id": f"faq:{stable_hash(q_clean + '|' + a_clean)}",
                "category": "faq",
                "question": q_clean,
                "answer": a_clean,
                "tags": sorted(set(tags)),
                "keywords": gen_keywords(q_clean + " " + a_clean),
                "source": "Частые вопросы"
            })
    return entries

# ── Сохранение ───────────────────────────────────────────────────────────────
def save_json(name: str, data: List[Dict]):
    out = OUT_DIR / name
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✔ {out} создан ({len(data)} записей)")

# ── Главный сценарий ─────────────────────────────────────────────────────────
def main():
    texts = {k: docx_to_text(p) for k, p in FILES.items()}

    save_json("structured_rooms.json",    build_rooms(texts["rooms"]))
    save_json("structured_concept.json",  build_concept(texts["concept"]))
    save_json("structured_contacts.json", build_contacts(texts["contacts"]))
    save_json("structured_hotel.json",    build_hotel(texts["hotel"]))
    save_json("structured_loyalty.json",  build_loyalty(texts["loyalty"]))
    save_json("structured_faq.json",      build_faq(texts["faq"]))

if __name__ == "__main__":
    main()
