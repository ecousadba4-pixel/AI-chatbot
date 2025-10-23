import os
import re
import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta, SA

# Сессии пользователей
user_sessions = {}

# ---------- Парсинг естественных выражений дат ----------
def parse_natural_date(user_input):
    text = user_input.lower().strip()
    today = datetime.today()

    # Завтра
    if "завтра" in text:
        return today + timedelta(days=1), None

    # Послезавтра
    if "послезавтра" in text:
        return today + timedelta(days=2), None

    # Эти выходные
    if "на выходных" in text or "эти выходные" in text:
        next_saturday = today + relativedelta(weekday=SA(+1))
        return next_saturday, 2  # по умолчанию 2 ночи

    # Следующие выходные
    if "следующ" in text and "выходных" in text:
        next_saturday = today + relativedelta(weekday=SA(+2))
        return next_saturday, 2

    # Через неделю
    if "через неделю" in text:
        return today + timedelta(days=7), None

    # Через месяц
    if "через месяц" in text:
        return today + relativedelta(months=1), None

    # Через несколько дней
    match = re.search(r"через\s+(\d+)\s+д", text)
    if match:
        delta_days = int(match.group(1))
        return today + timedelta(days=delta_days), None

    # Формат YYYY-MM-DD
    try:
        return datetime.strptime(text, "%Y-%m-%d"), None
    except ValueError:
        return None, None

# ---------- Запрос стоимости через Shelter API ----------
def get_room_price_from_shelter(date_from, date_to, adults, kids_ages):
    try:
        response = requests.post(
            "https://pms.frontdesk24.ru/api/online/getVariants",
            headers={
                "Content-Type": "application/json",
                "token": os.getenv("SHELTER_TOKEN")
            },
            json={
                "dateFrom": date_from,
                "dateTo": date_to,
                "rooms": [{"adults": adults, "kidsAges": kids_ages}]
            },
            timeout=15
        )

        data = response.json()
        if "variants" in data and data["variants"]:
            best = data["variants"][0]
            name = best.get("name", "Номер")
            price = best.get("priceRub", 0)
            return f"{name} стоит {price} руб. за период {date_from}–{date_to}."
        return "Нет доступных номеров на выбранные даты."
    except Exception as e:
        return f"Ошибка при обращении к Shelter API: {e}"

# ---------- Логика поэтапного диалога ----------
def handle_price_dialog(user_id, user_input):
    if user_id not in user_sessions:
        user_sessions[user_id] = {"step": 0, "info": {}}
    session = user_sessions[user_id]

    # Этап 0 – определение запроса о цене
    if session["step"] == 0:
        if any(kw in user_input.lower() for kw in ["цена", "стоимость", "сколько стоит", "забронировать", "номер"]):
            session["step"] = 1
            return {"answer": "Уточните, пожалуйста: с какой даты вы хотите заехать?", "mode": "booking"}
        else:
            return None

    # Этап 1 – обработка даты заезда
    elif session["step"] == 1:
        parsed_date, default_nights = parse_natural_date(user_input)
        if parsed_date:
            session["info"]["date_from"] = parsed_date.strftime("%Y-%m-%d")
            if default_nights:
                session["info"]["date_to"] = (
                    parsed_date + timedelta(days=default_nights)
                ).strftime("%Y-%m-%д")
                session["step"] = 3
                return {"answer": "Сколько взрослых будет проживать?", "mode": "booking"}
            else:
                session["step"] = 2
                return {"answer": "На сколько дней планируете проживание?", "mode": "booking"}
        else:
            return {"answer": "Введите дату в формате ГГГГ-ММ-ДД или фразой вроде 'на выходных', 'завтра'.", "mode": "booking"}

    # Этап 2 – количество дней проживания
    elif session["step"] == 2:
        try:
            days = int(user_input)
            start_date = datetime.strptime(session["info"]["date_from"], "%Y-%m-%d")
            session["info"]["date_to"] = (start_date + timedelta(days=days)).strftime("%Y-%m-%d")
            session["step"] = 3
            return {"answer": "Сколько взрослых будет проживать?", "mode": "booking"}
        except ValueError:
            return {"answer": "Введите количество дней проживания числом, например 2.", "mode": "booking"}

    # Этап 3 – количество взрослых
    elif session["step"] == 3:
        try:
            adults = int(user_input)
            session["info"]["adults"] = adults
            session["step"] = 4
            return {"answer": "Есть ли дети? Укажите их возрасты через запятую (например: 5, 9) или напишите 'нет'.", "mode": "booking"}
        except ValueError:
            return {"answer": "Введите количество взрослых числом.", "mode": "booking"}

    # Этап 4 – дети и расчет стоимости
    elif session["step"] == 4:
        kids_ages = []
        if user_input.lower() != "нет":
            try:
                kids_ages = [int(a.strip()) for a in user_input.split(",") if a.strip().isdigit()]
            except Exception:
                return {"answer": "Возраст детей укажите числами через запятую.", "mode": "booking"}

        info = session["info"]
        price = get_room_price_from_shelter(
            info["date_from"], info["date_to"], info["adults"], kids_ages
        )
        user_sessions[user_id] = {"step": 0, "info": {}}
        return {"answer": price, "mode": "booking"}
