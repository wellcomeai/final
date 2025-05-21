import os
from typing import Dict, List, Any, ClassVar
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    REALTIME_WS_URL: str = "wss://realtime.api.openai.com/v1/audio/speech"
    DEFAULT_VOICE: str = "alloy"
    DEFAULT_SYSTEM_MESSAGE: str = "Ты полезный голосовой ассистент. Отвечай кратко и по делу на русском языке."
    
    # Аннотация типа для DEFAULT_ASSISTANT_CONFIG или использование ClassVar
    DEFAULT_ASSISTANT_CONFIG: ClassVar[Dict[str, Any]] = {
        "name": "Тестовый голосовой ассистент",
        "voice": "alloy",
        "language": "ru",
        "system_prompt": DEFAULT_SYSTEM_MESSAGE,
        "functions": []  # Здесь можно добавить разрешенные функции
    }

    class Config:
        # Настройки модели для Pydantic 2.x
        env_file = ".env"
        extra = "ignore"

settings = Settings()
