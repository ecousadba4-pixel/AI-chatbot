from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from typing import Any, Iterable, Optional

import pymorphy3
import requests
from dateutil.relativedelta import SA, relativedelta

# ===============================
# Настройка логирования
# ===============================
logger = logging.getLogger(__name__)

# Морфологический анализатор используем для распознавания намерений
_morph = pymorphy3.MorphAnalyzer()

# Ключевые леммы, которые сигнализируют о запросе цены / бронирования
PRICE_KEYWORD_LEMMAS = {
    "цена",
    "стоимость",
    "забронировать",
    "бронирование",
    "бронь",
    "номер",
    "проживание",
    "ночь",
}

# Отдельные фразы без морфологии
PRICE_KEYWORD_PHRASES = [
    "сколько стоит",
]

def _normalize_words(text: str) -> set[str]:
    """Вернуть множество лемм слов в тексте."""

    tokens = re.findall(r"[а-яёa-z]+", text.lower())
    normalized: set[str] = set()

    for token in tokens:
        try:
            parsed = _morph.parse(token)
        except Exception:  # pragma: no cover - защита от редких сбоев pymorphy
            parsed = None

        if parsed:
            normalized.add(parsed[0].normal_form)
        else:
            normalized.add(token)

    return normalized

# ===============================
# Константы
# ===============================
MAX_ADULTS = 11
MAX_TOTAL_GUESTS = 11
MAX_STAY_DAYS = 30
MIN_STAY_DAYS = 1

class DialogStep(IntEnum):
    """Шаги диалога бронирования."""

    INTENT_DETECTION = 0
    CHECKIN_DATE = 1
    NIGHTS_COUNT = 2
    ADULTS_COUNT = 3
    CHILDREN_INFO = 4


@dataclass
class BookingSession:
    """Сериализуемая сессия диалога."""

    user_id: str
    redis_client: Any
    step: DialogStep = DialogStep.INTENT_DETECTION
    info: dict[str, Any] = field(default_factory=dict)
    last_activity: datetime = field(default_factory=datetime.now)

    _TTL_SECONDS: int = 3600
    _REDIS_PREFIX: str = "booking_session:"

    @property
    def redis_key(self) -> str:
        return f"{self._REDIS_PREFIX}{self.user_id}"

    @classmethod
    def load(cls, user_id: str, redis_client: Any) -> "BookingSession":
        raw = redis_client.get(f"{cls._REDIS_PREFIX}{user_id}")
        if not raw:
            return cls(user_id=user_id, redis_client=redis_client)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Не удалось распарсить сохранённую сессию, создаём новую")
            redis_client.delete(f"{cls._REDIS_PREFIX}{user_id}")
            return cls(user_id=user_id, redis_client=redis_client)

        last_activity_str: Optional[str] = data.get("last_activity")
        last_activity = (
            datetime.fromisoformat(last_activity_str)
            if isinstance(last_activity_str, str)
            else datetime.now()
        )

        step_value = data.get("step", DialogStep.INTENT_DETECTION)
        try:
            step = DialogStep(step_value)
        except ValueError:
            step = DialogStep.INTENT_DETECTION

        info = data.get("info") or {}
        if not isinstance(info, dict):
            info = {}

        return cls(
            user_id=user_id,
            redis_client=redis_client,
            step=step,
            info=info,
            last_activity=last_activity,
        )

    def touch(self) -> None:
        self.last_activity = datetime.now()

    def save(self) -> None:
        payload = {
            "step": int(self.step),
            "info": self.info,
            "last_activity": self.last_activity.isoformat(),
        }
        self.redis_client.setex(
            self.redis_key,
            self._TTL_SECONDS,
            json.dumps(payload, default=str),
        )

    def delete(self) -> None:
        self.redis_client.delete(self.redis_key)

# ===============================
# Парсинг естественных выражений даты
# ===============================
def parse_natural_date(user_input: str) -> tuple[Optional[datetime], Optional[int]]:
    """Парсинг естественных выражений даты с улучшенной логикой."""
    text = user_input.lower().strip()
    today = datetime.today()

    # Точные совпадения
    if "завтра" in text:
        return today + timedelta(days=1), None
    if "послезавтра" in text:
        return today + timedelta(days=2), None
    if "на выходных" in text or "эти выходные" in text:
        next_saturday = today + relativedelta(weekday=SA(+1))
        return next_saturday, 2
    if "следующ" in text and "выходных" in text:
        next_saturday = today + relativedelta(weekday=SA(+2))
        return next_saturday, 2
    if "через неделю" in text:
        return today + timedelta(days=7), None
    if "через месяц" in text:
        return today + relativedelta(months=1), None

    # Парсинг "через N дней"
    match = re.search(r"через\s+(\d+)\s+д", text)
    if match:
        return today + timedelta(days=int(match.group(1))), None

    # Парсинг конкретных дат в разных форматах
    date_formats = [
        "%Y-%m-%d",
        "%d.%m.%Y",
        "%d/%m/%Y",
        "%d %m %Y"
    ]

    for fmt in date_formats:
        try:
            return datetime.strptime(text, fmt), None
        except ValueError:
            continue

    return None, None

# ===============================
# Форматирование даты для пользователя
# ===============================
def format_date_russian(date_str: str) -> str:
    """Форматирует дату в русский формат."""
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    months = ["января", "февраля", "марта", "апреля", "мая", "июня",
              "июля", "августа", "сентября", "октября", "ноября", "декабря"]
    return f"{date_obj.day} {months[date_obj.month - 1]}"

# ===============================
# Извлечение числа из текста
# ===============================
def extract_number(text: str) -> Optional[int]:
    """Извлекает число из текста."""
    match = re.search(r'\d+', text)
    return int(match.group()) if match else None

# ===============================
# Валидация дат бронирования
# ===============================
def validate_dates(date_from: str, date_to: str) -> tuple[bool, str]:
    """Валидация дат бронирования."""
    try:
        checkin = datetime.strptime(date_from, "%Y-%m-%d")
        checkout = datetime.strptime(date_to, "%Y-%m-%d")
        today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

        if checkin < today:
            return False, "Дата заезда не может быть в прошлом"
        if checkout <= checkin:
            return False, "Дата выезда должна быть после даты заезда"
        if (checkout - checkin).days > MAX_STAY_DAYS:
            return False, f"Максимальный срок проживания - {MAX_STAY_DAYS} дней"
        if (checkout - checkin).days < MIN_STAY_DAYS:
            return False, f"Минимальный срок проживания - {MIN_STAY_DAYS} день"

        return True, ""
    except ValueError as e:
        logger.error(f"Ошибка валидации дат: {e}")
        return False, "Неверный формат даты"

# ===============================
# Валидация количества гостей
# ===============================
def validate_guests(adults: int, kids_ages: Iterable[int]) -> tuple[bool, str]:
    """Валидация количества гостей."""
    if adults < 1:
        return False, "Должен быть хотя бы один взрослый"
    if adults > MAX_ADULTS:
        return False, f"Максимальное количество взрослых - {MAX_ADULTS}"

    total_guests = adults + len(kids_ages)
    if total_guests > MAX_TOTAL_GUESTS:
        return False, f"Максимальное количество гостей в номере - {MAX_TOTAL_GUESTS}"

    # Проверка возраста детей
    for age in kids_ages:
        if age < 0:
            return False, "Возраст ребенка не может быть отрицательным"
        if age >= 12:
            return False, "Дети 12 лет и старше считаются взрослыми"

    return True, ""

# ===============================
# Основная функция обращения к Shelter
# ===============================
def get_room_price_from_shelter(
    date_from: str,
    date_to: str,
    adults: int,
    kids_ages: Iterable[int],
) -> str:
    """Получение цен на номера из Shelter API."""
    try:
        kids_ages_list = list(kids_ages)

        logger.info(
            "Запрос к Shelter API: %s - %s, взрослые: %s, дети: %s",
            date_from,
            date_to,
            adults,
            kids_ages_list,
        )

        # Валидация дат
        is_valid, error_msg = validate_dates(date_from, date_to)
        if not is_valid:
            return error_msg

        # Валидация гостей
        is_valid, error_msg = validate_guests(adults, kids_ages)
        if not is_valid:
            return error_msg

        payload = {
            "token": os.getenv("SHELTER_TOKEN"),
            "currency": "",
            "dateFrom": date_from,
            "dateTo": date_to,
            "language": "ru",
            "onlyRostourismProgram": 0,
            "rooms": [{"adults": adults}],
            "roomsOnly": None,
            "promocode": None
        }

        if kids_ages_list:
            payload["rooms"][0]["kidsAges"] = ",".join(str(age) for age in kids_ages_list)

        response = requests.post(
            "https://pms.frontdesk24.ru/api/online/getVariants",
            headers={
                "Content-Type": "application/json",
                "token": os.getenv("SHELTER_TOKEN")
            },
            json=payload,
            timeout=15
        )

        if response.status_code != 200:
            logger.error(f"Shelter API error: {response.status_code} - {response.text}")
            return "Извините, произошла ошибка при получении цен. Пожалуйста, попробуйте позже."

        data = response.json()
        variants = data.get("variants", [])

        if not variants:
            return "К сожалению, на выбранные даты нет доступных номеров."

        # Сортируем по цене и берем три самых дешевых предложения
        sorted_variants = sorted(variants, key=lambda x: x.get("priceRub", 0))
        results = []

        for v in sorted_variants[:3]:
            name = v.get("name", "Номер")
            price = v.get("priceRub", 0)
            tariff = v.get("tariffName", "")
            breakfast = "с завтраком" if "завтрак" in tariff.lower() else "без завтрака"

            # Форматируем цену с разделителями тысяч
            formatted_price = f"{price:,}".replace(",", " ")
            results.append(f"• {name} — {formatted_price}₽ за весь период ({breakfast})")

        nights = (datetime.strptime(date_to, "%Y-%m-%d") - datetime.strptime(date_from, "%Y-%m-%d")).days
        date_from_formatted = format_date_russian(date_from)
        date_to_formatted = format_date_russian(date_to)

        header = f"🏨 Доступные номера на {nights} ночей ({date_from_formatted} - {date_to_formatted}):\n\n"

        return header + "\n".join(results)

    except requests.exceptions.Timeout:
        logger.error("Shelter API timeout")
        return "Извините, сервис бронирования временно недоступен. Пожалуйста, попробуйте позже."
    except requests.exceptions.ConnectionError:
        logger.error("Shelter API connection error")
        return "Извините, нет соединения с сервисом бронирования. Пожалуйста, проверьте интернет-соединение."
    except Exception as e:
        logger.error("Ошибка при обращении к Shelter API: %s", e)
        return "Извините, произошла непредвиденная ошибка. Пожалуйста, попробуйте позже или свяжитесь с администратором."

class BookingDialog:
    """Пошаговый обработчик диалога о бронировании."""

    def __init__(self, user_id: str, user_input: str, redis_client: Any) -> None:
        self.text = user_input.strip()
        self.session = BookingSession.load(user_id=user_id, redis_client=redis_client)
        self.session.touch()

    @property
    def redis(self) -> Any:
        return self.session.redis_client

    def _respond(self, message: str) -> dict[str, str]:
        self.session.save()
        return {"answer": message, "mode": "booking"}

    def _finish(self, message: str) -> dict[str, str]:
        self.session.delete()
        return {"answer": message, "mode": "booking"}

    def _is_booking_intent(self) -> bool:
        normalized_words = _normalize_words(self.text)
        has_keyword = bool(PRICE_KEYWORD_LEMMAS & normalized_words)
        has_phrase = any(phrase in self.text.lower() for phrase in PRICE_KEYWORD_PHRASES)
        return has_keyword or has_phrase

    def _handle_intent(self) -> Optional[dict[str, str]]:
        if not self._is_booking_intent():
            return None

        self.session.step = DialogStep.CHECKIN_DATE
        return self._respond(
            "Отлично! Помогу узнать цены на номера. Введите дату заезда "
            "(например '2025-10-24', 'завтра' или 'на выходных')."
        )

    def _handle_checkin(self) -> dict[str, str]:
        parsed_date, default_nights = parse_natural_date(self.text)
        if not parsed_date:
            return self._respond(
                "Пожалуйста, введите дату в формате ГГГГ-ММ-ДД или используйте "
                "выражения: 'завтра', 'послезавтра', 'на выходных', 'через неделю'."
            )

        self.session.info["date_from"] = parsed_date.strftime("%Y-%m-%d")

        if default_nights:
            self.session.info["date_to"] = (
                parsed_date + timedelta(days=default_nights)
            ).strftime("%Y-%m-%d")
            self.session.step = DialogStep.ADULTS_COUNT
            date_from_formatted = format_date_russian(self.session.info["date_from"])
            return self._respond(
                f"Отлично! Вы выбрали заезд {date_from_formatted} на {default_nights} ночей. "
                f"Сколько взрослых будет проживать? (максимум {MAX_ADULTS})"
            )

        self.session.step = DialogStep.NIGHTS_COUNT
        date_from_formatted = format_date_russian(self.session.info["date_from"])
        return self._respond(
            f"Заезд {date_from_formatted}. На сколько ночей планируется проживание? "
            f"(максимум {MAX_STAY_DAYS})"
        )

    def _handle_nights(self) -> dict[str, str]:
        nights = extract_number(self.text)
        if nights is None:
            return self._respond(
                "Пожалуйста, введите количество ночей числом (например: 2, 3, 7)."
            )

        if nights < MIN_STAY_DAYS:
            return self._respond(
                f"Количество ночей должно быть не менее {MIN_STAY_DAYS}."
            )
        if nights > MAX_STAY_DAYS:
            return self._respond(
                f"Максимальный срок проживания - {MAX_STAY_DAYS} ночей."
            )

        start_date = datetime.strptime(self.session.info["date_from"], "%Y-%m-%d")
        self.session.info["date_to"] = (
            start_date + timedelta(days=nights)
        ).strftime("%Y-%m-%d")
        self.session.step = DialogStep.ADULTS_COUNT

        date_from_formatted = format_date_russian(self.session.info["date_from"])
        date_to_formatted = format_date_russian(self.session.info["date_to"])

        return self._respond(
            f"Отлично! {nights} ночей с {date_from_formatted} по {date_to_formatted}. "
            f"Сколько взрослых будет проживать? (максимум {MAX_ADULTS})"
        )

    def _handle_adults(self) -> dict[str, str]:
        adults = extract_number(self.text)
        if adults is None:
            return self._respond("Пожалуйста, введите количество взрослых числом.")

        if adults < 1:
            return self._respond("Должен быть хотя бы один взрослый.")
        if adults > MAX_ADULTS:
            return self._respond(
                f"Максимальное количество взрослых - {MAX_ADULTS}."
            )

        self.session.info["adults"] = adults
        max_children = MAX_TOTAL_GUESTS - adults

        if max_children <= 0:
            result = get_room_price_from_shelter(
                self.session.info["date_from"],
                self.session.info["date_to"],
                adults,
                [],
            )
            return self._finish(result)

        self.session.step = DialogStep.CHILDREN_INFO
        return self._respond(
            "Есть ли дети? Укажите их возрасты через запятую (например: 5, 9) или "
            f"напишите 'нет'. Максимум можно добавить {max_children} детей."
        )

    def _parse_children_ages(self) -> list[int]:
        if self.text.lower() in {"нет", "детей нет", "без детей"}:
            return []

        ages: list[int] = []
        for chunk in self.text.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if not chunk.isdigit():
                raise ValueError("Возраст ребёнка должен быть числом")
            ages.append(int(chunk))
        return ages

    def _handle_children(self) -> dict[str, str]:
        try:
            kids_ages = self._parse_children_ages()
        except ValueError as exc:
            logger.error("Ошибка парсинга возрастов детей: %s", exc)
            return self._respond(
                "Пожалуйста, укажите возрасты детей числами через запятую "
                "(например: 5, 9) или напишите 'нет'."
            )

        if not all(0 <= age <= 11 for age in kids_ages):
            return self._respond(
                "Возраст детей должен быть от 0 до 11 лет. Укажите правильные "
                "возрасты через запятую или напишите 'нет'."
            )

        adults = self.session.info["adults"]
        if adults + len(kids_ages) > MAX_TOTAL_GUESTS:
            return self._respond(
                f"Слишком много гостей. У вас {adults} взрослых и {len(kids_ages)} детей. "
                f"Максимум гостей в номере - {MAX_TOTAL_GUESTS}. Укажите меньше детей "
                "или напишите 'нет'."
            )

        self.session.info["kids_ages"] = kids_ages
        result = get_room_price_from_shelter(
            self.session.info["date_from"],
            self.session.info["date_to"],
            adults,
            kids_ages,
        )
        return self._finish(result)

    def handle(self) -> Optional[dict[str, str]]:
        try:
            if self.session.step == DialogStep.INTENT_DETECTION:
                return self._handle_intent()
            if self.session.step == DialogStep.CHECKIN_DATE:
                return self._handle_checkin()
            if self.session.step == DialogStep.NIGHTS_COUNT:
                return self._handle_nights()
            if self.session.step == DialogStep.ADULTS_COUNT:
                return self._handle_adults()
            if self.session.step == DialogStep.CHILDREN_INFO:
                return self._handle_children()

            logger.warning("Неизвестный шаг диалога: %s", self.session.step)
            return self._finish(
                "Извините, произошла ошибка при обработке запроса. Пожалуйста, начните заново."
            )
        except Exception as exc:  # pragma: no cover - защита от непредвиденных сбоев
            logger.error("Ошибка в handle_price_dialog: %s", exc)
            self.session.delete()
            return {
                "answer": "Извините, произошла ошибка при обработке запроса. "
                "Пожалуйста, начните заново.",
                "mode": "booking",
            }


def handle_price_dialog(user_id: str, user_input: str, redis_client: Any) -> Optional[dict[str, str]]:
    """Точка входа для обработки диалога по стоимости проживания."""

    dialog = BookingDialog(user_id=user_id, user_input=user_input, redis_client=redis_client)
    return dialog.handle()

