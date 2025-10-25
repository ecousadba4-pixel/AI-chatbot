import os
import requests
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta, SA
import re
import logging
import pymorphy3

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


def _normalize_words(text):
    """Возвращает множество нормальных форм слов из текста."""

    tokens = re.findall(r"[а-яёa-z]+", text.lower())
    normalized = set()

    for token in tokens:
        try:
            parsed = _morph.parse(token)
            if parsed:
                normalized.add(parsed[0].normal_form)
            else:
                normalized.add(token)
        except Exception:
            normalized.add(token)

    return normalized

# ===============================
# Константы
# ===============================
MAX_ADULTS = 11
MAX_TOTAL_GUESTS = 11
MAX_STAY_DAYS = 30
MIN_STAY_DAYS = 1

# ===============================
# Хелперы для Redis
# ===============================
def get_session(user_id, redis_client):
    """Получить сессию из Redis."""
    data = redis_client.get(f"booking_session:{user_id}")
    if data:
        session = json.loads(data)
        # Конвертируем строку last_activity обратно в datetime
        session["last_activity"] = datetime.fromisoformat(session["last_activity"])
        return session
    return {"step": 0, "info": {}, "last_activity": datetime.now()}

def save_session(user_id, session_data, redis_client):
    """Сохранить сессию в Redis с TTL 1 час."""
    session_data["last_activity"] = session_data["last_activity"].isoformat()
    redis_client.setex(
        f"booking_session:{user_id}",
        3600,  # 1 час
        json.dumps(session_data, default=str)
    )

def delete_session(user_id, redis_client):
    """Удалить сессию."""
    redis_client.delete(f"booking_session:{user_id}")

# ===============================
# Парсинг естественных выражений даты
# ===============================
def parse_natural_date(user_input):
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
def format_date_russian(date_str):
    """Форматирует дату в русский формат."""
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    months = ["января", "февраля", "марта", "апреля", "мая", "июня",
              "июля", "августа", "сентября", "октября", "ноября", "декабря"]
    return f"{date_obj.day} {months[date_obj.month - 1]}"

# ===============================
# Извлечение числа из текста
# ===============================
def extract_number(text):
    """Извлекает число из текста."""
    match = re.search(r'\d+', text)
    return int(match.group()) if match else None

# ===============================
# Валидация дат бронирования
# ===============================
def validate_dates(date_from, date_to):
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
def validate_guests(adults, kids_ages):
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
def get_room_price_from_shelter(date_from, date_to, adults, kids_ages):
    """Получение цен на номера из Shelter API."""
    try:
        logger.info(f"Запрос к Shelter API: {date_from} - {date_to}, взрослые: {adults}, дети: {kids_ages}")

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

        if kids_ages:
            payload["rooms"][0]["kidsAges"] = ",".join(str(age) for age in kids_ages)

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
        logger.error(f"Ошибка при обращении к Shelter API: {e}")
        return "Извините, произошла непредвиденная ошибка. Пожалуйста, попробуйте позже или свяжитесь с администратором."

# ===============================
# Пошаговая логика диалога
# ===============================
def handle_price_dialog(user_id, user_input, redis_client):
    """Обработка диалога для получения цен на номера."""
    try:
        # Получаем сессию из Redis
        session = get_session(user_id, redis_client)
        session["last_activity"] = datetime.now()

        # Шаг 0: Определение намерения
        if session["step"] == 0:
            normalized_words = _normalize_words(user_input)

            has_keyword = bool(PRICE_KEYWORD_LEMMAS & normalized_words)
            has_phrase = any(phrase in user_input.lower() for phrase in PRICE_KEYWORD_PHRASES)

            if has_keyword or has_phrase:
                session["step"] = 1
                save_session(user_id, session, redis_client)
                return {
                    "answer": "Отлично! Помогу узнать цены на номера. Введите дату заезда (например '2025-10-24', 'завтра' или 'на выходных').",
                    "mode": "booking"
                }
            return None

        # Шаг 1: Получение даты заезда
        if session["step"] == 1:
            parsed_date, default_nights = parse_natural_date(user_input)
            if parsed_date:
                session["info"]["date_from"] = parsed_date.strftime("%Y-%m-%d")

                if default_nights:
                    # Для выражений типа "на выходных" автоматически ставим 2 ночи
                    session["info"]["date_to"] = (parsed_date + timedelta(days=default_nights)).strftime("%Y-%m-%d")
                    session["step"] = 3
                    save_session(user_id, session, redis_client)

                    date_from_formatted = format_date_russian(session['info']['date_from'])
                    return {
                        "answer": f"Отлично! Вы выбрали заезд {date_from_formatted} на {default_nights} ночей. Сколько взрослых будет проживать? (максимум {MAX_ADULTS})",
                        "mode": "booking",
                    }
                else:
                    session["step"] = 2
                    save_session(user_id, session, redis_client)

                    date_from_formatted = format_date_russian(session['info']['date_from'])
                    return {
                        "answer": f"Заезд {date_from_formatted}. На сколько ночей планируется проживание? (максимум {MAX_STAY_DAYS})",
                        "mode": "booking",
                    }

            return {
                "answer": "Пожалуйста, введите дату в формате ГГГГ-ММ-ДД или используйте выражения: 'завтра', 'послезавтра', 'на выходных', 'через неделю'.",
                "mode": "booking",
            }

        # Шаг 2: Получение количества ночей
        if session["step"] == 2:
            nights = extract_number(user_input)
            if nights is None:
                return {"answer": "Пожалуйста, введите количество ночей числом (например: 2, 3, 7).", "mode": "booking"}

            if nights < MIN_STAY_DAYS:
                return {"answer": f"Количество ночей должно быть не менее {MIN_STAY_DAYS}.", "mode": "booking"}
            if nights > MAX_STAY_DAYS:
                return {"answer": f"Максимальный срок проживания - {MAX_STAY_DAYS} ночей.", "mode": "booking"}

            start_date = datetime.strptime(session["info"]["date_from"], "%Y-%m-%d")
            session["info"]["date_to"] = (start_date + timedelta(days=nights)).strftime("%Y-%m-%d")
            session["step"] = 3
            save_session(user_id, session, redis_client)

            date_from_formatted = format_date_russian(session['info']['date_from'])
            date_to_formatted = format_date_russian(session['info']['date_to'])

            return {
                "answer": f"Отлично! {nights} ночей с {date_from_formatted} по {date_to_formatted}. Сколько взрослых будет проживать? (максимум {MAX_ADULTS})",
                "mode": "booking",
            }

        # Шаг 3: Получение количества взрослых
        if session["step"] == 3:
            adults = extract_number(user_input)
            if adults is None:
                return {"answer": "Пожалуйста, введите количество взрослых числом.", "mode": "booking"}

            if adults < 1:
                return {"answer": "Должен быть хотя бы один взрослый.", "mode": "booking"}
            if adults > MAX_ADULTS:
                return {"answer": f"Максимальное количество взрослых - {MAX_ADULTS}.", "mode": "booking"}

            session["info"]["adults"] = adults
            session["step"] = 4
            save_session(user_id, session, redis_client)

            max_kids = MAX_TOTAL_GUESTS - adults
            if max_kids > 0:
                return {
                    "answer": f"Есть ли дети? Укажите их возрасты через запятую (например: 5, 9) или напишите 'нет'. Максимум можно добавить {max_kids} детей.",
                    "mode": "booking"
                }
            else:
                session["step"] = 5
                save_session(user_id, session, redis_client)
                info = session["info"]
                result = get_room_price_from_shelter(info["date_from"], info["date_to"], info["adults"], [])
                delete_session(user_id, redis_client)
                return {"answer": result, "mode": "booking"}

        # Шаг 4: Получение инфрмации о детях
        if session["step"] == 4:
            kids_ages = []
            if user_input.lower().strip() not in ["нет", "детей нет", "без детей"]:
                try:
                    kids_ages = [int(a.strip()) for a in user_input.split(",") if a.strip().isdigit()]

                    # Валидация возраста детей
                    valid_ages = all(0 <= age <= 11 for age in kids_ages)

                    if not valid_ages:
                        return {
                            "answer": "Возраст детей должен быть от 0 до 11 лет. Укажите правильные возрасты через запятую или напишите 'нет'.",
                            "mode": "booking",
                        }

                    # Проверка общего количества гостей
                    total_guests = session["info"]["adults"] + len(kids_ages)
                    if total_guests > MAX_TOTAL_GUESTS:
                        return {
                            "answer": f"Слишком много гостей. У вас {session['info']['adults']} взрослых и {len(kids_ages)} детей. Максимум гостей в номере - {MAX_TOTAL_GUESTS}. Укажите меньше детей или напишите 'нет'.",
                            "mode": "booking",
                        }

                except Exception as e:
                    logger.error(f"Ошибка парсинга возрастов детей: {e}")
                    return {
                        "answer": "Пожалуйста, укажите возрасты детей числами через запятую (например: 5, 9) или напишите 'нет'.",
                        "mode": "booking",
                    }

            # Переходим к финальному шагу
            session["step"] = 5
            session["info"]["kids_ages"] = kids_ages
            save_session(user_id, session, redis_client)

            info = session["info"]
            result = get_room_price_from_shelter(info["date_from"], info["date_to"], info["adults"], kids_ages)
            delete_session(user_id, redis_client)

            return {"answer": result, "mode": "booking"}

    except Exception as e:
        logger.error(f"Ошибка в handle_price_dialog: {e}")
        # Сброс сессии при ошибке
        delete_session(user_id, redis_client)

        return {
            "answer": "Извините, произошла ошибка при обработке запроса. Пожалуйста, начните заново.",
            "mode": "booking"
        }


