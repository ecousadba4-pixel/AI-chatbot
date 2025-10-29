"""Пошаговый диалог бронирования с обращением к Shelter API."""
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


LOGGER = logging.getLogger("chatbot.price_dialog")

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
PRICE_KEYWORD_PHRASES = ("сколько стоит",)

MAX_ADULTS = 11
MAX_TOTAL_GUESTS = 11
MAX_STAY_DAYS = 30
MIN_STAY_DAYS = 1

SHELTER_URL = "https://pms.frontdesk24.ru/api/online/getVariants"
SHELTER_TOKEN_ENV = "SHELTER_TOKEN"
SHELTER_TIMEOUT = 15
DATE_FORMAT = "%Y-%m-%d"

_SESSIONS: dict[str, "BookingSession"] = {}


def _cleanup_expired_sessions(now: datetime) -> None:
    for key, session in list(_SESSIONS.items()):
        if (now - session.last_activity).total_seconds() > session._ttl_seconds:
            _SESSIONS.pop(key, None)



def _normalize_words(text: str, morph: pymorphy3.MorphAnalyzer) -> set[str]:
    tokens = re.findall(r"[а-яёa-z]+", text.lower())
    lemmas: set[str] = set()
    for token in tokens:
        try:
            parsed = morph.parse(token)
        except Exception:  # pragma: no cover - защита от редких сбоев pymorphy
            parsed = None
        lemmas.add(parsed[0].normal_form if parsed else token)
    return lemmas


class DialogStep(IntEnum):
    INTENT_DETECTION = 0
    CHECKIN_DATE = 1
    NIGHTS_COUNT = 2
    ADULTS_COUNT = 3
    CHILDREN_INFO = 4


@dataclass(slots=True)
class BookingSession:
    user_id: str
    step: DialogStep = field(default=DialogStep.INTENT_DETECTION)
    info: dict[str, Any] = field(default_factory=dict)
    last_activity: datetime = field(default_factory=datetime.now)

    _ttl_seconds: int = field(default=3600, init=False, repr=False)

    @classmethod
    def load(cls, user_id: str) -> "BookingSession":
        now = datetime.now()
        _cleanup_expired_sessions(now)
        session = _SESSIONS.get(user_id)
        if session is None:
            session = cls(user_id=user_id)
            _SESSIONS[user_id] = session
        return session

    def touch(self) -> None:
        self.last_activity = datetime.now()

    def save(self) -> None:
        self.last_activity = datetime.now()
        _SESSIONS[self.user_id] = self

    def delete(self) -> None:
        _SESSIONS.pop(self.user_id, None)


def parse_natural_date(user_input: str) -> tuple[Optional[datetime], Optional[int]]:
    text = user_input.lower().strip()
    today = datetime.today()

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

    match = re.search(r"через\s+(\d+)\s+д", text)
    if match:
        return today + timedelta(days=int(match.group(1))), None

    date_formats = ["%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d %m %Y"]
    for fmt in date_formats:
        try:
            return datetime.strptime(text, fmt), None
        except ValueError:
            continue

    return None, None


def format_date_russian(date_str: str) -> str:
    date_obj = datetime.strptime(date_str, DATE_FORMAT)
    months = [
        "января",
        "февраля",
        "марта",
        "апреля",
        "мая",
        "июня",
        "июля",
        "августа",
        "сентября",
        "октября",
        "ноября",
        "декабря",
    ]
    return f"{date_obj.day} {months[date_obj.month - 1]}"


def extract_number(text: str) -> Optional[int]:
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


def validate_dates(date_from: str, date_to: str) -> tuple[bool, str]:
    try:
        checkin = datetime.strptime(date_from, DATE_FORMAT)
        checkout = datetime.strptime(date_to, DATE_FORMAT)
        today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

        if checkin < today:
            return False, "Дата заезда не может быть в прошлом"
        if checkout <= checkin:
            return False, "Дата выезда должна быть позже даты заезда"

        nights = (checkout - checkin).days
        if nights < MIN_STAY_DAYS:
            return False, f"Минимальный срок проживания - {MIN_STAY_DAYS} день"
        if nights > MAX_STAY_DAYS:
            return False, f"Максимальный срок проживания - {MAX_STAY_DAYS} ночей"

        return True, ""
    except ValueError as exc:
        LOGGER.error("Ошибка валидации дат: %s", exc)
        return False, "Неверный формат даты"


def validate_guests(adults: int, kids_ages: Iterable[int]) -> tuple[bool, str]:
    if adults < 1:
        return False, "Должен быть хотя бы один взрослый"
    if adults > MAX_ADULTS:
        return False, f"Максимальное количество взрослых - {MAX_ADULTS}"

    kids = list(kids_ages)
    total_guests = adults + len(kids)
    if total_guests > MAX_TOTAL_GUESTS:
        return False, f"Максимальное количество гостей в номере - {MAX_TOTAL_GUESTS}"

    for age in kids:
        if age < 0:
            return False, "Возраст ребенка не может быть отрицательным"
        if age >= 12:
            return False, "Дети 12 лет и старше считаются взрослыми"

    return True, ""


def _load_shelter_token() -> Optional[str]:
    token = (os.getenv(SHELTER_TOKEN_ENV) or "").strip()
    return token or None


@dataclass(slots=True)
class ShelterVariant:
    name: str
    price_rub: int
    tariff: str

    def format_line(self) -> str:
        formatted_price = f"{self.price_rub:,}".replace(",", " ")
        breakfast = "с завтраком" if "завтрак" in self.tariff.lower() else "без завтрака"
        return f"• {self.name} — {formatted_price}₽ за весь период ({breakfast})"


def _build_shelter_payload(
    *,
    token: str,
    date_from: str,
    date_to: str,
    adults: int,
    kids_ages: Iterable[int],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "token": token,
        "currency": "",
        "dateFrom": date_from,
        "dateTo": date_to,
        "language": "ru",
        "onlyRostourismProgram": 0,
        "rooms": [{"adults": adults}],
        "roomsOnly": None,
        "promocode": None,
    }

    kids_list = list(kids_ages)
    if kids_list:
        payload["rooms"][0]["kidsAges"] = ",".join(str(age) for age in kids_list)

    return payload


def get_room_price_from_shelter(
    date_from: str,
    date_to: str,
    adults: int,
    kids_ages: Iterable[int],
) -> str:
    kids_ages_list = list(kids_ages)

    is_valid, error_msg = validate_dates(date_from, date_to)
    if not is_valid:
        return error_msg

    is_valid, error_msg = validate_guests(adults, kids_ages_list)
    if not is_valid:
        return error_msg

    token = _load_shelter_token()
    if not token:
        LOGGER.error("Не задан токен Shelter API (%s)", SHELTER_TOKEN_ENV)
        return "Сервис бронирования временно недоступен. Пожалуйста, свяжитесь с администратором."

    payload = _build_shelter_payload(
        token=token,
        date_from=date_from,
        date_to=date_to,
        adults=adults,
        kids_ages=kids_ages_list,
    )

    headers = {"Content-Type": "application/json", "token": token}

    try:
        response = requests.post(
            SHELTER_URL,
            headers=headers,
            json=payload,
            timeout=SHELTER_TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        LOGGER.error("Shelter API timeout")
        return "Извините, сервис бронирования временно недоступен. Пожалуйста, попробуйте позже."
    except requests.exceptions.ConnectionError:
        LOGGER.error("Shelter API connection error")
        return "Извините, нет соединения с сервисом бронирования. Пожалуйста, проверьте интернет-соединение."
    except requests.RequestException as exc:
        LOGGER.error("Shelter API error: %s", exc)
        return "Извините, произошла ошибка при получении цен. Пожалуйста, попробуйте позже."

    try:
        data = response.json()
    except ValueError as exc:
        LOGGER.error("Некорректный JSON от Shelter API: %s", exc)
        return "Извините, произошла ошибка при обработке ответа сервиса бронирования."

    variants = data.get("variants") or []
    if not variants:
        return "К сожалению, на выбранные даты нет доступных номеров."

    sorted_variants: list[ShelterVariant] = []
    for variant in variants:
        price_raw = variant.get("priceRub", 0)
        try:
            price_value = int(price_raw)
        except (TypeError, ValueError):
            price_value = 0

        sorted_variants.append(
            ShelterVariant(
                name=variant.get("name", "Номер"),
                price_rub=price_value,
                tariff=variant.get("tariffName", ""),
            )
        )

    sorted_variants.sort(key=lambda item: item.price_rub)

    nights = (datetime.strptime(date_to, DATE_FORMAT) - datetime.strptime(date_from, DATE_FORMAT)).days
    date_from_formatted = format_date_russian(date_from)
    date_to_formatted = format_date_russian(date_to)

    header = f"🏨 Доступные номера на {nights} ночей ({date_from_formatted} - {date_to_formatted}):\n\n"
    lines = [variant.format_line() for variant in sorted_variants[:3]]

    return header + "\n".join(lines)


class BookingDialog:
    """Пошаговый обработчик диалога о бронировании."""

    def __init__(
        self,
        user_id: str,
        user_input: str,
        morph: pymorphy3.MorphAnalyzer,
    ) -> None:
        self.text = user_input.strip()
        self.morph = morph
        self.session = BookingSession.load(user_id=user_id)
        self.session.touch()

    def _respond(self, message: str) -> dict[str, str]:
        self.session.save()
        return {"answer": message, "mode": "booking"}

    def _finish(self, message: str) -> dict[str, str]:
        self.session.delete()
        return {"answer": message, "mode": "booking"}

    def _is_booking_intent(self) -> bool:
        normalized_words = _normalize_words(self.text, self.morph)
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

        self.session.info["date_from"] = parsed_date.strftime(DATE_FORMAT)

        if default_nights:
            self.session.info["date_to"] = (
                parsed_date + timedelta(days=default_nights)
            ).strftime(DATE_FORMAT)
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
            return self._respond("Пожалуйста, введите количество ночей числом (например: 2, 3, 7).")

        if nights < MIN_STAY_DAYS:
            return self._respond(f"Количество ночей должно быть не менее {MIN_STAY_DAYS}.")
        if nights > MAX_STAY_DAYS:
            return self._respond(f"Максимальный срок проживания - {MAX_STAY_DAYS} ночей.")

        start_date = datetime.strptime(self.session.info["date_from"], DATE_FORMAT)
        self.session.info["date_to"] = (start_date + timedelta(days=nights)).strftime(DATE_FORMAT)
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
            return self._respond(f"Максимальное количество взрослых - {MAX_ADULTS}.")

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
        lower = self.text.lower()
        if lower in {"нет", "детей нет", "без детей"}:
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
            LOGGER.error("Ошибка парсинга возрастов детей: %s", exc)
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
            step = self.session.step
            if step is DialogStep.INTENT_DETECTION:
                return self._handle_intent()
            if step is DialogStep.CHECKIN_DATE:
                return self._handle_checkin()
            if step is DialogStep.NIGHTS_COUNT:
                return self._handle_nights()
            if step is DialogStep.ADULTS_COUNT:
                return self._handle_adults()
            if step is DialogStep.CHILDREN_INFO:
                return self._handle_children()

            LOGGER.warning("Неизвестный шаг диалога: %s", step)
            return self._finish(
                "Извините, произошла ошибка при обработке запроса. Пожалуйста, начните заново."
            )
        except Exception:  # pragma: no cover - защита от непредвиденных сбоев
            LOGGER.exception("Ошибка в handle_price_dialog")
            self.session.delete()
            return {
                "answer": "Извините, произошла ошибка при обработке запроса. "
                "Пожалуйста, начните заново.",
                "mode": "booking",
            }


def handle_price_dialog(
    user_id: str,
    user_input: str,
    morph: pymorphy3.MorphAnalyzer,
) -> Optional[dict[str, str]]:
    dialog = BookingDialog(
        user_id=user_id,
        user_input=user_input,
        morph=morph,
    )
    return dialog.handle()


def clear_booking_session(user_id: str) -> None:
    _SESSIONS.pop(user_id, None)


__all__ = [
    "DialogStep",
    "BookingSession",
    "ShelterVariant",
    "handle_price_dialog",
    "clear_booking_session",
    "get_room_price_from_shelter",
    "parse_natural_date",
    "validate_dates",
    "validate_guests",
]
