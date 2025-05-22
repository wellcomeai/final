from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import uvicorn
import os
import logging
from pathlib import Path

from handler import handle_websocket_connection
from config import settings, validate_config

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Создаем папку static, если её нет
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Создаем FastAPI приложение
app = FastAPI(
    title="Voice Assistant API",
    description="API для голосового ассистента с поддержкой OpenAI Realtime",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Подключаем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    """Событие запуска приложения"""
    logger.info("Запуск приложения Voice Assistant")
    
    # Проверяем конфигурацию
    if not validate_config():
        logger.warning("Обнаружены проблемы в конфигурации")
    
    # Выводим информацию о настройках
    logger.info(f"OpenAI API ключ: {'установлен' if settings.OPENAI_API_KEY else 'НЕ УСТАНОВЛЕН'}")
    logger.info(f"Порт: {settings.PORT}")
    logger.info(f"Хост: {settings.HOST}")
    logger.info(f"Модель: {settings.DEFAULT_MODEL}")
    logger.info(f"Голос по умолчанию: {settings.DEFAULT_VOICE}")

@app.on_event("shutdown")
async def shutdown_event():
    """Событие остановки приложения"""
    logger.info("Остановка приложения Voice Assistant")

@app.websocket("/ws")
async def websocket_endpoint_default(websocket: WebSocket):
    """WebSocket эндпоинт для подключения с дефолтным ассистентом"""
    await handle_websocket_connection(websocket, "default")

@app.websocket("/ws/{assistant_id}")
async def websocket_endpoint(websocket: WebSocket, assistant_id: str):
    """WebSocket эндпоинт для подключения к конкретному ассистенту"""
    await handle_websocket_connection(websocket, assistant_id)

@app.get("/")
async def root():
    """Корневой эндпоинт - перенаправляет на веб-интерфейс"""
    return RedirectResponse(url="/static/index.html")

@app.get("/config-page")
async def config_page():
    """Страница настроек"""
    return RedirectResponse(url="/static/config.html")

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy",
        "service": "voice-assistant",
        "version": "1.0.0",
        "openai_configured": bool(settings.OPENAI_API_KEY)
    }

@app.get("/config")
async def get_config():
    """Получение публичной конфигурации"""
    return {
        "model": settings.DEFAULT_MODEL,
        "voice": settings.DEFAULT_VOICE,
        "sample_rate": settings.AUDIO_SAMPLE_RATE,
        "channels": settings.AUDIO_CHANNELS,
        "functions_enabled": settings.FUNCTIONS_ENABLED,
        "max_message_size": settings.MAX_MESSAGE_SIZE
    }

@app.get("/assistants")
async def list_assistants():
    """Список доступных ассистентов"""
    return {
        "assistants": [
            {
                "id": "default",
                "name": "Стандартный ассистент",
                "description": "Базовый голосовой ассистент"
            },
            {
                "id": "demo",
                "name": "Демо ассистент", 
                "description": "Демонстрационный ассистент для тестирования"
            },
            {
                "id": "webhook_assistant",
                "name": "Вебхук ассистент",
                "description": "Ассистент с поддержкой вебхуков"
            }
        ]
    }

@app.get("/functions")
async def list_functions():
    """Список доступных функций"""
    from config import get_function_definitions
    
    functions = get_function_definitions()
    return {
        "functions": [
            {
                "name": name,
                "description": func_def["description"],
                "parameters": func_def["parameters"]
            }
            for name, func_def in functions.items()
        ]
    }

# Обработчик ошибок
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Глобальный обработчик ошибок"""
    logger.error(f"Необработанная ошибка: {exc}", exc_info=True)
    return {
        "error": "Internal server error",
        "message": "Произошла внутренняя ошибка сервера"
    }

if __name__ == "__main__":
    # Проверяем конфигурацию перед запуском
    if not validate_config():
        logger.error("Критические ошибки в конфигурации. Остановка.")
        exit(1)
    
    # Запуск сервера
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=False,  # В продакшене должно быть False
        workers=settings.WORKERS
    )
