import os
from typing import Dict, List, Any, ClassVar
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    # Исправляем URL для WebSocket и REST API
    REALTIME_SESSION_URL: str = "https://api.openai.com/v1/realtime/sessions"
    DEFAULT_VOICE: str = "alloy"
    DEFAULT_SYSTEM_MESSAGE: str = "Ты полезный голосовой ассистент. Отвечай кратко и по делу на русском языке."
    
    # Добавляем настройки модели для Realtime API
    DEFAULT_MODEL: str = "gpt-4o-realtime-preview"
    
    # Аннотация типа для DEFAULT_ASSISTANT_CONFIG
    DEFAULT_ASSISTANT_CONFIG: ClassVar[Dict[str, Any]] = {
        "name": "Тестовый голосовой ассистент",
        "voice": "alloy",
        "language": "ru",
        "system_prompt": DEFAULT_SYSTEM_MESSAGE,
        "model": DEFAULT_MODEL,
        "functions": []  # Здесь можно добавить разрешенные функции
    }

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
