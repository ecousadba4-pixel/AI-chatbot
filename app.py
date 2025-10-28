"""Точка входа Flask-приложения."""
from __future__ import annotations

import os

from chatbot import create_app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
