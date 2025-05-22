import os
from typing import Dict, List, Any, ClassVar
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # OpenAI настройки
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    
    # URL для OpenAI Realtime WebSocket API
    REALTIME_WS_URL: str = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
    
    # Голосовые настройки
    DEFAULT_VOICE: str = "alloy"
    DEFAULT_SYSTEM_MESSAGE: str = "Ты полезный голосовой ассистент. Отвечай кратко и по делу на русском языке. ОБЯЗАТЕЛЬНО отвечай голосом, а не только текстом. Будь дружелюбным и говори естественно."
    
    # Модель для Realtime API
    DEFAULT_MODEL: str = "gpt-4o-realtime-preview-2024-10-01"
    
    # Настройки аудио
    AUDIO_SAMPLE_RATE: int = 24000
    AUDIO_CHANNELS: int = 1
    AUDIO_SAMPLE_WIDTH: int = 2  # 16-bit
    
    # Настройки VAD (Voice Activity Detection)
    VAD_THRESHOLD: float = 0.25
    VAD_PREFIX_PADDING_MS: int = 200
    VAD_SILENCE_DURATION_MS: int = 300
    
    # Настройки функций
    FUNCTIONS_ENABLED: bool = True
    WEBHOOK_TIMEOUT: int = 10  # секунды
    
    # Аннотация типа для DEFAULT_ASSISTANT_CONFIG
    DEFAULT_ASSISTANT_CONFIG: ClassVar[Dict[str, Any]] = {
        "name": "Голосовой ассистент",
        "voice": "alloy",
        "language": "ru",
        "system_prompt": "Ты полезный голосовой ассистент. Отвечай кратко и по делу на русском языке.",
        "model": "gpt-4o-realtime-preview-2024-10-01",
        "functions": []  # Здесь можно добавить разрешенные функции
    }
    
    # Настройки безопасности и лимитов
    MAX_MESSAGE_SIZE: int = 15 * 1024 * 1024  # 15 MB
    WEBSOCKET_PING_INTERVAL: int = 30  # секунды
    WEBSOCKET_PING_TIMEOUT: int = 120  # секунды
    WEBSOCKET_CLOSE_TIMEOUT: int = 15  # секунды
    CONNECTION_TIMEOUT: int = 30  # секунды
    
    # Настройки переподключений
    RECONNECT_ATTEMPTS: int = 3
    RECONNECT_DELAY: int = 3  # секунды
    
    # Настройки логирования
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Настройки для продакшена
    HOST: str = "0.0.0.0"
    PORT: int = int(os.environ.get("PORT", 8000))
    WORKERS: int = 1
    
    # CORS настройки
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    class Config:
        env_file = ".env"
        extra = "ignore"

# Создаем глобальный экземпляр настроек
settings = Settings()

# Функции-помощники для работы с конфигурацией
def get_assistant_config(assistant_id: str = "default") -> Dict[str, Any]:
    """
    Возвращает конфигурацию ассистента по ID.
    В продвинутой версии это будет загружаться из базы данных.
    """
    # В базовой версии возвращаем дефолтную конфигурацию
    config = settings.DEFAULT_ASSISTANT_CONFIG.copy()
    
    # Можно добавить специфичные настройки для разных ассистентов
    if assistant_id == "demo":
        config.update({
            "name": "Демо ассистент",
            "system_prompt": "Ты демонстрационный голосовой ассистент. Приветствуй пользователей и помогай им тестировать функциональность.",
        })
    elif assistant_id == "webhook_assistant":
        config.update({
            "name": "Вебхук ассистент",
            "system_prompt": "Ты ассистент с поддержкой вебхуков. URL вебхука: https://example.com/webhook",
            "functions": [
                {
                    "name": "send_webhook",
                    "enabled": True
                }
            ]
        })
    
    return config

def get_function_definitions() -> Dict[str, Dict[str, Any]]:
    """
    Возвращает определения всех доступных функций.
    """
    return {
        "send_webhook": {
            "name": "send_webhook",
            "description": "Отправляет данные на webhook URL для интеграции с внешними системами",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL вебхука для отправки данных"
                    },
                    "event": {
                        "type": "string", 
                        "description": "Тип события или действия"
                    },
                    "data": {
                        "type": "object",
                        "description": "Данные для отправки",
                        "properties": {},
                        "additionalProperties": True
                    }
                },
                "required": ["url", "event"]
            }
        }
    }

def validate_config() -> bool:
    """
    Проверяет корректность конфигурации.
    """
    if not settings.OPENAI_API_KEY:
        print("Предупреждение: OPENAI_API_KEY не установлен")
        return False
    
    if settings.AUDIO_SAMPLE_RATE not in [16000, 24000, 48000]:
        print(f"Предупреждение: Неподдерживаемая частота дискретизации: {settings.AUDIO_SAMPLE_RATE}")
        return False
    
    return True

# Проверяем конфигурацию при импорте
if __name__ == "__main__":
    if validate_config():
        print("Конфигурация корректна")
    else:
        print("Обнаружены проблемы в конфигурации")
