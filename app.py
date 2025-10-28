"""REST API чат-бота для усадьбы "Четыре сезона"."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

import requests
from flask import Flask, current_app, jsonify, request
from flask_cors import CORS

from amvera import (
    AmveraError,
    build_payload,
    cache_key,
    ensure_token,
    extract_answer,
    log_error,
    perform_request,
)
from config import Settings
from price_dialog import handle_price_dialog
from rag import SearchResult, encode, normalize_text, search_all_collections
from services import Dependencies, create_dependencies


LOGGER = logging.getLogger("chatbot")
CANCEL_COMMANDS = {"отмена", "сброс", "начать заново", "стоп", "reset"}
ERROR_MESSAGE = "Извините, не удалось получить ответ. Пожалуйста, попробуйте позже."


@dataclass(frozen=True)
class AppContainer:
    """Собранные настройки и зависимости приложения."""

    settings: Settings
    dependencies: Dependencies
    collections: tuple[str, ...]


def configure_logging() -> None:
    """Настроить базовое логирование один раз за время жизни процесса."""

    if logging.getLogger().handlers:
        # Логирование уже настроено окружением (например, gunicorn)
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _filter_existing_collections(
    client, requested: Iterable[str], fallback: Iterable[str]
) -> tuple[str, ...]:
    try:
        response = client.get_collections()
        available = {collection.name for collection in response.collections}
    except Exception as exc:  # pragma: no cover - сетевой сбой
        LOGGER.warning("Не удалось получить список коллекций из Qdrant: %s", exc)
        return tuple(requested)

    if not available:
        LOGGER.warning("В Qdrant не найдено ни одной коллекции")
        return tuple(requested)

    requested_list = list(dict.fromkeys(requested))
    filtered = [name for name in requested_list if name in available]
    if filtered:
        missing = sorted(name for name in requested_list if name not in available)
        if missing:
            LOGGER.warning(
                "Пропускаем отсутствующие коллекции: %s",
                ", ".join(missing),
            )
        return tuple(filtered)

    fallback_candidates = [name for name in fallback if name in available]
    if not fallback_candidates:
        fallback_candidates = sorted(available)
    LOGGER.warning(
        "Ни одна из запрошенных коллекций не найдена. Используем fallback: %s",
        ", ".join(fallback_candidates),
    )
    return tuple(fallback_candidates)


def _collect_public_endpoints(app: Flask) -> list[str]:
    collected: dict[str, None] = {}
    for rule in app.url_map.iter_rules():
        if rule.endpoint == "static" or rule.rule == "/":
            continue
        collected.setdefault(rule.rule, None)

    default_order = [
        "/api/chat",
        "/api/debug/qdrant",
        "/api/debug/redis",
        "/api/debug/search",
        "/api/debug/amvera",
        "/api/debug/model",
        "/api/debug/status",
        "/health",
    ]

    ordered: list[str] = []
    for path in default_order:
        if path in collected:
            ordered.append(path)
            collected.pop(path)

    ordered.extend(sorted(collected.keys()))
    return ordered


def _json_reply(session_id: str, message: str, **extra: Any):
    payload = {"response": message, "session_id": session_id}
    payload.update(extra)
    return jsonify(payload)


def _build_context(results: list[SearchResult]) -> str:
    return "\n\n".join(result.text for result in results)


@dataclass(slots=True)
class ChatResponse:
    """Результат обработки пользовательского запроса."""

    message: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatResponder:
    """Инкапсулирует бизнес-логику обработки сообщений."""

    container: AppContainer

    def handle(self, session_id: str, question: str) -> ChatResponse:
        LOGGER.info("Вопрос [%s]: %s", session_id[:8], question)

        lower_question = question.lower()
        if lower_question in CANCEL_COMMANDS:
            self._clear_booking_session(session_id)
            return ChatResponse("Диалог сброшен. Чем могу помочь?")

        booking_result = handle_price_dialog(
            session_id,
            question,
            self.container.dependencies.redis,
            self.container.dependencies.morph,
        )
        if booking_result:
            extra = {k: v for k, v in booking_result.items() if k != "answer"}
            return ChatResponse(booking_result["answer"], extra)

        normalized = self.normalize(question)
        LOGGER.debug("Нормализованный запрос: %s", normalized)

        search_results, query_embedding, backend = self.perform_semantic_search(
            normalized,
            limit=5,
        )

        LOGGER.debug("Размер эмбеддинга запроса: %s", len(query_embedding))
        if backend == "qdrant":
            LOGGER.info(
                "Поиск в Qdrant по коллекциям: %s",
                ", ".join(self.container.collections),
            )
        else:
            local_index = self.container.dependencies.local_index
            document_count = local_index.document_count if local_index else 0
            LOGGER.info(
                "Поиск в локальном индексе (%s документов)",
                document_count,
            )

        if not search_results:
            LOGGER.info("Ничего не найдено ни в одной коллекции")
            return ChatResponse(
                "Извините, не нашёл информации по вашему вопросу. "
                "Попробуйте переформулировать или свяжитесь с администратором.",
            )

        LOGGER.debug("Топ-результаты: %s", search_results[:3])

        context = _build_context(search_results[:3])
        if not context.strip():
            LOGGER.warning("Контекст пуст после извлечения payload['text']")
            return ChatResponse(
                "Извините, не удалось сформировать ответ. "
                "Попробуйте переформулировать вопрос.",
            )

        LOGGER.debug("Итоговый контекст длиной %s символов", len(context))

        answer = self._generate_response(context, question)
        LOGGER.info("Ответ сгенерирован: %s", answer[:100].replace("\n", " "))

        debug_info = {
            "top_collection": search_results[0].collection,
            "top_score": search_results[0].score,
            "results_count": len(search_results),
            "embedding_dim": len(query_embedding),
            "search_backend": backend,
        }
        return ChatResponse(answer, {"debug_info": debug_info})

    def normalize(self, text: str) -> str:
        return normalize_text(text, self.container.dependencies.morph)

    def perform_semantic_search(
        self, normalized: str, *, limit: int
    ) -> tuple[list[SearchResult], list[float], str]:
        deps = self.container.dependencies
        if deps.local_index is not None:
            results, query_vector = deps.local_index.search(normalized, limit=limit)
            return results, query_vector, "local"

        query_embedding = encode(normalized, deps.embedding_model)
        results = search_all_collections(
            deps.qdrant,
            self.container.collections,
            query_embedding,
            limit=limit,
        )
        return results, query_embedding, "qdrant"

    def _generate_response(self, context: str, question: str) -> str:
        deps = self.container.dependencies
        settings = self.container.settings

        redis_key = cache_key(context, question)

        cached = deps.redis.get(redis_key)
        if cached:
            LOGGER.info("Ответ для вопроса %s получен из кэша Redis", question)
            return cached

        try:
            token = ensure_token(settings)
        except AmveraError as exc:
            LOGGER.warning("%s", exc)
            return ERROR_MESSAGE

        payload = build_payload(settings.amvera_model, context, question)

        try:
            response = perform_request(settings, token, payload, timeout=60)
        except requests.RequestException as exc:
            LOGGER.warning("Не удалось выполнить запрос к Amvera API: %s", exc)
            return ERROR_MESSAGE

        if not response.ok:
            log_error(response)
            return ERROR_MESSAGE

        try:
            data = response.json()
        except ValueError:
            LOGGER.warning("Не удалось распарсить ответ Amvera как JSON")
            return ERROR_MESSAGE

        try:
            answer = extract_answer(data)
        except AmveraError as exc:
            LOGGER.warning("%s", exc)
            return ERROR_MESSAGE

        try:
            deps.redis.setex(redis_key, 3600, answer)
        except Exception as exc:  # pragma: no cover - сбой Redis не критичен
            LOGGER.warning("Не удалось сохранить ответ в Redis: %s", exc)
        else:
            LOGGER.debug("Ответ сохранён в кэш Redis под ключом %s", redis_key)

        return answer

    def _clear_booking_session(self, session_id: str) -> None:
        try:
            self.container.dependencies.redis.delete(
                f"booking_session:{session_id}"
            )
        except Exception as exc:  # pragma: no cover - сбой Redis не критичен
            LOGGER.warning("Не удалось очистить сессию бронирования: %s", exc)


def _get_container() -> AppContainer:
    container = current_app.config.get("container")
    if not isinstance(container, AppContainer):  # pragma: no cover - защитная проверка
        raise RuntimeError("Конфигурация приложения не инициализирована")
    return container


def _log_startup_information(container: AppContainer) -> None:
    settings = container.settings
    deps = container.dependencies

    if deps.qdrant is not None:
        LOGGER.info(
            "Connected to Qdrant at %s:%s (https=%s)",
            settings.qdrant_host,
            settings.qdrant_port,
            settings.qdrant_https,
        )
    elif deps.local_index is not None:
        LOGGER.info(
            "Qdrant недоступен, используется локальный индекс (%s коллекций)",
            len(deps.local_index.collections),
        )
    LOGGER.info(
        "Connected to Redis at %s:%s",
        settings.redis_host,
        settings.redis_port,
    )
    embedding_dim = deps.embedding_model.get_sentence_embedding_dimension()
    LOGGER.info("Embedding dimension: %s", embedding_dim)
    if deps.local_index is not None:
        resolved_from = "local_index"
        LOGGER.info(
            "Локальный индекс: %s документов", deps.local_index.document_count
        )
    else:
        resolved_from = getattr(
            deps.embedding_model, "_resolved_from", settings.embedding_model
        )
    LOGGER.info("Источник модели эмбеддингов: %s", resolved_from)
    LOGGER.info(
        "Amvera GPT endpoint: %s (model=%s)",
        settings.amvera_url,
        settings.amvera_model,
    )


def create_app(
    *,
    settings: Settings | None = None,
    dependencies: Dependencies | None = None,
) -> Flask:
    """Сконструировать и сконфигурировать экземпляр Flask-приложения."""

    configure_logging()

    resolved_settings = settings or Settings.from_env()
    resolved_dependencies = dependencies or create_dependencies(resolved_settings)
    if resolved_dependencies.local_index is not None:
        collections = resolved_dependencies.local_index.collections
    else:
        collections = _filter_existing_collections(
            resolved_dependencies.qdrant,
            resolved_settings.default_collections,
            resolved_settings.default_collections,
        )

    container = AppContainer(
        settings=resolved_settings,
        dependencies=resolved_dependencies,
        collections=collections,
    )

    app = Flask(__name__)
    CORS(app)
    app.config["container"] = container

    _log_startup_information(container)

    register_routes(app)
    return app


def register_routes(app: Flask) -> None:
    """Зарегистрировать HTTP-маршруты на переданном приложении."""

    @app.route("/api/chat", methods=["POST"])
    def chat() -> Any:  # noqa: D401 - функция возвращает JSON-ответ
        container = _get_container()
        data = request.get_json(silent=True) or {}
        question = data.get("message", "").strip()
        session_id = data.get("session_id") or os.urandom(16).hex()

        if not question:
            return _json_reply(session_id, "Пожалуйста, введите вопрос.")

        responder = ChatResponder(container)
        response = responder.handle(session_id, question)
        return _json_reply(session_id, response.message, **response.extra)

    @app.route("/api/debug/qdrant", methods=["GET"])
    def debug_qdrant() -> Any:
        container = _get_container()
        if container.dependencies.qdrant is None:
            return jsonify(
                {
                    "status": "disabled",
                    "message": "Qdrant не используется, активирован локальный индекс",
                }
            )
        try:
            collections = container.dependencies.qdrant.get_collections().collections
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)})

        return jsonify(
            {
                "status": "ok",
                "collections": [collection.name for collection in collections],
            }
        )

    @app.route("/api/debug/redis", methods=["GET"])
    def debug_redis() -> Any:
        container = _get_container()
        try:
            container.dependencies.redis.ping()
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)})

        return jsonify({"status": "ok", "message": "Redis connection active"})

    @app.route("/api/debug/search", methods=["POST"])
    def debug_search() -> Any:
        container = _get_container()
        responder = ChatResponder(container)

        data = request.get_json(silent=True) or {}
        question = data.get("message", "").strip()

        if not question:
            return jsonify({"error": "message required"}), 400

        normalized = responder.normalize(question)
        results, vector, backend = responder.perform_semantic_search(
            normalized,
            limit=10,
        )

        return jsonify(
            {
                "question": question,
                "normalized": normalized,
                "embedding_dim": len(vector),
                "search_backend": backend,
                "results": [
                    {
                        "collection": result.collection,
                        "score": result.score,
                        "text_preview": result.text[:200],
                    }
                    for result in results
                ],
            }
        )

    @app.route("/api/debug/amvera", methods=["GET"], strict_slashes=False)
    def debug_amvera() -> Any:
        container = _get_container()
        settings = container.settings

        prompt = request.args.get("prompt", "Привет! Ответь 'ok'.")
        model_name = request.args.get("model") or settings.amvera_model

        try:
            token = ensure_token(settings)
        except AmveraError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 503

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "text": "Ты — простая проверка доступа к API."},
                {"role": "user", "text": prompt},
            ],
        }

        try:
            response = perform_request(settings, token, payload, timeout=30)
        except requests.RequestException as exc:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Не удалось выполнить запрос: {exc}",
                    }
                ),
                502,
            )

        if response.ok:
            try:
                response_json = response.json()
            except ValueError:
                response_json = {"raw": response.text}
            return jsonify(
                {
                    "status": "ok",
                    "model": model_name,
                    "prompt": prompt,
                    "response": response_json,
                }
            )

        log_error(response)
        try:
            error_body = response.json()
        except ValueError:
            error_body = {"raw": response.text}
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Amvera API вернул ошибку",
                    "http_status": response.status_code,
                    "details": error_body,
                }
            ),
            response.status_code,
        )

    @app.route("/api/debug/model", methods=["GET"])
    def debug_model() -> Any:
        container = _get_container()
        deps = container.dependencies
        settings = container.settings
        return jsonify(
            {
                "model": settings.embedding_model,
                "embedding_dimension": deps.embedding_model.get_sentence_embedding_dimension(),
            }
        )

    @app.route("/api/debug/status", methods=["GET"])
    def debug_status() -> Any:
        container = _get_container()
        deps = container.dependencies
        settings = container.settings

        services_state: dict[str, Any] = {}

        if deps.qdrant is None:
            services_state["qdrant"] = {
                "status": "disabled",
                "message": "Используется локальный индекс",
            }
        else:
            try:
                deps.qdrant.get_collections()
                services_state["qdrant"] = {
                    "status": "ok",
                    "host": settings.qdrant_host,
                    "port": settings.qdrant_port,
                }
            except Exception as exc:
                services_state["qdrant"] = {
                    "status": "error",
                    "message": str(exc),
                    "host": settings.qdrant_host,
                    "port": settings.qdrant_port,
                }

        try:
            deps.redis.ping()
            services_state["redis"] = {
                "status": "ok",
                "host": settings.redis_host,
                "port": settings.redis_port,
            }
        except Exception as exc:
            services_state["redis"] = {
                "status": "error",
                "message": str(exc),
                "host": settings.redis_host,
                "port": settings.redis_port,
            }

        try:
            services_state["embedding_model"] = {
                "status": "ok",
                "name": settings.embedding_model,
                "source": getattr(
                    deps.embedding_model,
                    "_resolved_from",
                    settings.embedding_model,
                )
                if deps.local_index is None
                else "local_index",
                "dimension": deps.embedding_model.get_sentence_embedding_dimension(),
            }
        except Exception as exc:
            services_state["embedding_model"] = {
                "status": "error",
                "name": settings.embedding_model,
                "message": str(exc),
            }

        services_state["amvera_gpt"] = {
            "status": "configured" if settings.amvera_url else "not_configured",
            "url": settings.amvera_url,
        }

        overall_status = (
            "ok"
            if all(service.get("status") != "error" for service in services_state.values())
            else "degraded"
        )

        return jsonify({"status": overall_status, "services": services_state})

    @app.route("/health")
    def health() -> Any:
        return "OK", 200

    @app.route("/")
    def home() -> Any:
        container = _get_container()
        deps = container.dependencies
        settings = container.settings
        return jsonify(
            {
                "status": "ok",
                "message": "Усадьба 'Четыре Сезона' - AI Assistant",
                "version": "4.0",
                "features": ["RAG", "Booking Dialog", "Redis Cache"],
                "embedding_model": settings.embedding_model,
                "embedding_dim": deps.embedding_model.get_sentence_embedding_dimension(),
                "embedding_source": getattr(
                    deps.embedding_model,
                    "_resolved_from",
                    settings.embedding_model,
                ),
                "search_backend": "local" if deps.local_index else "qdrant",
                "endpoints": _collect_public_endpoints(app),
            }
        )


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
