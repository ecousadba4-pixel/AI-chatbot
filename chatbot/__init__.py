"""Основной пакет чат-бота."""
from .web import AppContainer, ChatResponder, create_app, register_routes

__all__ = [
    "AppContainer",
    "ChatResponder",
    "create_app",
    "register_routes",
]
