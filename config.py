import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    REALTIME_WS_URL: str = "wss://realtime.api.openai.com/v1/audio/speech"
    DEFAULT_VOICE: str = "alloy"
    DEFAULT_SYSTEM_MESSAGE: str = "Ты полезный голосовой ассистент. Отвечай кратко и по делу на русском языке."
    
    # Настройки для ассистента по умолчанию
    DEFAULT_ASSISTANT_CONFIG = {
        "name": "Тестовый голосовой ассистент",
        "voice": "alloy",
        "language": "ru",
        "system_prompt": DEFAULT_SYSTEM_MESSAGE,
        "functions": []  # Здесь можно добавить разрешенные функции
    }

settings = Settings()
