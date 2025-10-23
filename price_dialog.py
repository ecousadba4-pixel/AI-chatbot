import os
import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta, SA
import re

# ===============================
# Хранение состояний пользователя
# ===============================
user_sessions = {}

# ===============================
# Парсинг естественных выражений даты
# ===============================
def parse_natural_date(user_input):
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
    try:
        return datetime.strptime(text, "%Y-%m-%d"), None
    except ValueError:
        return None, None

# ===============================
# Основная функция обращения к Shelter
# ===============================
def get_room_price_from_shelter(date_from, date_to, adults, kids_ages):
    try:
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

        data = response.json()
        variants = data.get("variants", [])
        if not variants:
            return "К сожалению, на выбранные даты нет доступных номеров."

        # берем только три самых дешевых предложения
        sorted_variants = sorted(variants, key=lambda x: x.get("priceRub", 0))
        results = []
        for v in sorted_variants[:3]:
            name = v.get("name", "Номер")
            price = v.get("priceRub", 0)
            tariff = v.get("tariffName", "")
            breakfast = "с завтраком" if "завтрак" in tariff.lower() else "без завтрака"
            results.append(f"{name} — {price}₽ за период проживания ({breakfast})")

        return "Доступные варианты:\n" + "\n".join(results)

    except Exception as e:
        return f"Ошибка при обращении к Shelter API: {e}"

# ===============================
# Пошаговая логика диалога
# ===============================
def handle_price_dialog(user_id, user_input):
    if user_id not in user_sessions:
        user_sessions[user_id] = {"step": 0, "info": {}}
    session = user_sessions[user_id]

    if session["step"] == 0:
        if any(kw in user_input.lower() for kw in ["цена", "стоимость", "номер", "сколько стоит", "забронировать"]):
            session["step"] = 1
            return {"answer": "Введите, пожалуйста, дату заезда (например 2025-10-24 или 'завтра').", "mode": "booking"}
        return None

    if session["step"] == 1:
        parsed_date, default_nights = parse_natural_date(user_input)
        if parsed_date:
            session["info"]["date_from"] = parsed_date.strftime("%Y-%m-%d")
            if default_nights:
                session["info"]["date_to"] = (parsed_date + timedelta(days=default_nights)).strftime("%Y-%m-%d")
                session["step"] = 3
                return {"answer": "Сколько взрослых будет проживать?", "mode": "booking"}
            else:
                session["step"] = 2
                return {"answer": "На сколько дней планируется проживание?", "mode": "booking"}
        return {"answer": "Введите дату в формате ГГГГ-ММ-ДД или фразой 'на выходных', 'завтра'.", "mode": "booking"}

    if session["step"] == 2:
        try:
            days = int(user_input)
            start_date = datetime.strptime(session["info"]["date_from"], "%Y-%m-%d")
            session["info"]["date_to"] = (start_date + timedelta(days=days)).strftime("%Y-%m-%д")
            session["step"] = 3
            return {"answer": "Сколько взрослых будет проживать?", "mode": "booking"}
        except ValueError:
            return {"answer": "Введите количество дней числом, например 2.", "mode": "booking"}

    if session["step"] == 3:
        try:
            adults = int(user_input)
            session["info"]["adults"] = adults
            session["step"] = 4
            return {"answer": "Есть ли дети? Укажите их возрасты через запятую (например: 5, 9) или напишите 'нет'.", "mode": "booking"}
        except ValueError:
            return {"answer": "Введите число взрослых числом.", "mode": "booking"}

    if session["step"] == 4:
        kids_ages = []
        if user_input.lower() != "нет":
            try:
                kids_ages = [int(a.strip()) for a in user_input.split(",") if a.strip().isdigit()]
            except Exception:
                return {"answer": "Возраст детей нужно указать числами через запятую.", "mode": "booking"}
        info = session["info"]
        result = get_room_price_from_shelter(info["date_from"], info["date_to"], info["adults"], kids_ages)
        user_sessions[user_id] = {"step": 0, "info": {}}
        return {"answer": result, "mode": "booking"}

