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
    # Нормализация Wi-Fi (лат/кирилл i/і, дефис/пробел/ничего)
    text = re.sub(r"\bW[iі][-\s_]*F[iі]\b", "Wi-Fi", text, flags=re.I)
    text = re.sub(r"\bWI[\s_-]*FII\b", "Wi-Fi", text, flags=re.I)
    text = re.sub(r"\bWi-?F\b", "Wi-Fi", text, flags=re.I)
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
PHONE_RAW_RE = re.compile(r"(?:\+7|8)\s*[\(\-]?\s*\d{3}\s*[\)\-]?\s*\d{3}[\s\-]?\d{2}[\s\-]?\d{2}")
DIGITS_RE = re.compile(r"\d+")

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

        entries.append({
            "id": f"rooms:{normalize_room_name(subcat) or abs(hash(title))}",
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
    entries = []
    phone_pat = PHONE_RAW_RE

    booking   = re.search(r"(брони\w*|заказ\w*|онлайн\s*бронир\w*).{0,100}(" + phone_pat.pattern + r")", text, flags=re.I | re.S)
    reception = re.search(r"(ресепшен|администратор|стойка).{0,80}(" + phone_pat.pattern + r")", text, flags=re.I)
    restaurant= re.search(r"(ресторан|кафе|бар).{0,80}(" + phone_pat.pattern + r")", text, flags=re.I)

    links = re.findall(r"(https?://[^\s]+)", text, flags=re.I)
    opening_hours = extract_opening_hours(text)
    has_wa = ("whatsapp" in text.lower()) or ("ватсап" in text.lower()) \
             or any("wa.me" in l.lower() or "whatsapp" in l.lower() for l in links)
    geo = extract_geo_from_yandex_links(links)

    def pack_contact(contact_id, ctype, title, phone_match):
        if not phone_match:
            return
        raw = phone_match.group(2)
        phones = [raw]
        phones_norm = list(filter(None, [normalize_phone_e164(raw)]))
        entries.append({
            "id": contact_id,
            "category": "contacts",
            "contact_type": ctype,
            "title": title,
            "phone": raw,
            "phones": phones,
            "phones_norm": phones_norm,
            "hours": opening_hours,
            "opening_hours": opening_hours,
            "whatsapp": has_wa,
            "links": links,
            "geo": geo,
            "text": f"{title}: {raw}" + (f"; часы: {opening_hours}" if opening_hours else ""),
            "keywords": ["контакты", ctype, "телефон"] + (["whatsapp"] if has_wa else []),
            "source": "Наши контакты"
        })

    pack_contact("contacts:booking", "booking", "Контакты для бронирования", booking)
    pack_contact("contacts:reception", "reception", "Телефон ресепшена", reception)
    pack_contact("contacts:restaurant", "restaurant", "Телефон ресторана", restaurant)

    social = [l for l in links if any(x in l.lower() for x in ["instagram.com", "t.me"])]
    if social:
        entries.append({
            "id": "contacts:social",
            "category": "contacts",
            "contact_type": "social",
            "title": "Социальные сети",
            "links": social,
            "text": " ; ".join(social),
            "keywords": ["соцсети"] + (["instagram"] if any("instagram" in l for l in social) else []) + (["telegram"] if any("t.me" in l for l in social) else []),
            "source": "Наши контакты"
        })

    m_dir = re.search(r"Как добраться на машине\s*(.+)$", text, flags=re.S)
    if m_dir:
        entries.append({
            "id": "contacts:directions_car",
            "category": "contacts",
            "contact_type": "directions",
            "title": "Как добраться на машине",
            "links": [l for l in links if "yandex" in l.lower()],
            "geo": geo,
            "text": re.sub(r"\s+", " ", m_dir.group(1)).strip(),
            "keywords": ["как добраться", "машина", "маршрут", "навигатор", "яндекс"],
            "source": "Наши контакты"
        })

    return entries

def build_hotel(text: str) -> List[Dict]:
    entries = []
    m_reviews = re.search(r"(https?://yandex[^\s]+)", text, flags=re.I)
    entries.append({
        "id": "hotel:about",
        "category": "hotel",
        "subcategory": "Общее описание",
        "title": "Общее описание отеля",
        "text": ("Мы — загородный эко-отель «Усадьба Четыре Сезона». В отеле ежегодно отдыхают более 3500 гостей."
                 + (f" Отзывы: {m_reviews.group(1)}." if m_reviews else "")),
        "keywords": ["эко-отель", "гости", "отзывы"],
        "source": "Описание отеля и доступных услуг"
    })

    for para in text.split("\n"):
        if "подойдет" in para.lower():
            entries.append({
                "id": "hotel:audience",
                "category": "hotel",
                "subcategory": "Кому подходит отдых",
                "title": "Кому подходит отдых",
                "text": para.strip(),
                "keywords": ["семьи", "пары", "друзья", "тимбилдинг"],
                "source": "Описание отеля и доступных услуг"
            })
            break

    m_loc = re.search(r"(Мы расположены[^\n\.]*\.)", text, flags=re.S)
    if m_loc:
        entries.append({
            "id": "hotel:location",
            "category": "hotel",
            "subcategory": "Расположение",
            "title": "Расположение",
            "text": re.sub(r"\s+", " ", m_loc.group(1)).strip(),
            "keywords": ["расположение", "Минское шоссе", "Можайский район", "деревня Власово", "100 км"],
            "source": "Описание отеля и доступных услуг"
        })

    m_terr = re.search(r"(Территория[^\n]*\d+\s*га[^\n]*)", text, flags=re.I)
    if m_terr:
        entries.append({
            "id": "hotel:territory",
            "category": "hotel",
            "subcategory": "Территория",
            "title": "Территория",
            "text": re.sub(r"\s+", " ", m_terr.group(1)).strip(),
            "keywords": ["территория", "га", "тихо", "огороженная"],
            "source": "Описание отеля и доступных услуг"
        })

    return entries

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
                "id": f"faq:{abs(hash(q_clean))}",
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
