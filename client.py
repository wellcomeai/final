import asyncio
import json
import uuid
import base64
import time
import websockets
from websockets.exceptions import ConnectionClosed
from typing import Dict, Any, Optional

from config import settings

class SimpleOpenAIClient:
    """Упрощенный клиент для работы с OpenAI Realtime API через WebSockets"""
    
    def __init__(self, api_key: Optional[str] = None, assistant_config: Optional[Dict] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.assistant_config = assistant_config or settings.DEFAULT_ASSISTANT_CONFIG
        self.client_id = str(uuid.uuid4())
        self.ws = None
        self.is_connected = False
        self.openai_url = settings.REALTIME_WS_URL
        self.session_id = str(uuid.uuid4())
        self.current_response_id = None
        
        # Простой список разрешенных функций
        self.enabled_functions = []
        if "functions" in self.assistant_config:
            self.enabled_functions = [f.get("name") for f in self.assistant_config["functions"] if f.get("name")]

    async def connect(self) -> bool:
        """Устанавливает WebSocket соединение с OpenAI Realtime API"""
        if not self.api_key:
            print("OpenAI API key не предоставлен")
            return False

        headers = [
            ("Authorization", f"Bearer {self.api_key}"),
            ("OpenAI-Beta", "realtime=v1"),
            ("User-Agent", "MiniVoiceAssistant/1.0")
        ]
        try:
            self.ws = await asyncio.wait_for(
                websockets.connect(
                    self.openai_url,
                    extra_headers=headers,
                    max_size=15*1024*1024,
                    ping_interval=30,
                    ping_timeout=120,
                    close_timeout=15
                ),
                timeout=30
            )
            self.is_connected = True
            print(f"Подключено к OpenAI для клиента {self.client_id}")

            # Получаем настройки из конфигурации
            voice = self.assistant_config.get("voice", settings.DEFAULT_VOICE)
            system_message = self.assistant_config.get("system_prompt", settings.DEFAULT_SYSTEM_MESSAGE)
            
            # Обновляем настройки сессии
            if not await self.update_session(voice=voice, system_message=system_message):
                print("Не удалось обновить настройки сессии")
                await self.close()
                return False

            return True
        except asyncio.TimeoutError:
            print(f"Таймаут подключения к OpenAI для клиента {self.client_id}")
            return False
        except Exception as e:
            print(f"Ошибка подключения к OpenAI: {e}")
            return False

    async def update_session(self, voice: str = None, system_message: str = None) -> bool:
        """Обновляет настройки сессии на стороне OpenAI Realtime API"""
        if not self.is_connected or not self.ws:
            print("Невозможно обновить сессию: нет подключения")
            return False
            
        voice = voice or settings.DEFAULT_VOICE
        system_message = system_message or settings.DEFAULT_SYSTEM_MESSAGE
            
        turn_detection = {
            "type": "server_vad",
            "threshold": 0.25,
            "prefix_padding_ms": 200,
            "silence_duration_ms": 300,
            "create_response": True,
        }
        
        # Упрощенные настройки без функций для минимальной версии
        payload = {
            "type": "session.update",
            "session": {
                "turn_detection": turn_detection,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "voice": voice,
                "instructions": system_message,
                "modalities": ["text", "audio"],
                "temperature": 0.7,
                "max_response_output_tokens": 500,
                "input_audio_transcription": {
                    "model": "whisper-1"
                }
            }
        }
        
        # Добавляем функции, если они есть
        if self.assistant_config.get("functions"):
            tools = []
            for func in self.assistant_config["functions"]:
                if "name" in func and "description" in func and "parameters" in func:
                    tools.append({
                        "type": "function",
                        "name": func["name"],
                        "description": func["description"],
                        "parameters": func["parameters"]
                    })
            
            if tools:
                payload["session"]["tools"] = tools
                payload["session"]["tool_choice"] = "auto"
                print(f"Добавлено {len(tools)} функций в сессию")
            
        try:
            await self.ws.send(json.dumps(payload))
            print(f"Настройки сессии отправлены (voice={voice})")
            return True
        except Exception as e:
            print(f"Ошибка отправки session.update: {e}")
            return False

    async def process_audio(self, audio_buffer: bytes) -> bool:
        """Обрабатывает и отправляет аудио данные в OpenAI API"""
        if not self.is_connected or not self.ws or not audio_buffer:
            return False
        try:
            data_b64 = base64.b64encode(audio_buffer).decode("utf-8")
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": data_b64,
                "event_id": f"audio_{time.time()}"
            }))
            return True
        except ConnectionClosed:
            print("Соединение закрыто при отправке аудио данных")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"Ошибка обработки аудио: {e}")
            return False

    async def commit_audio(self) -> bool:
        """Фиксирует аудио буфер, указывая, что пользователь закончил говорить"""
        if not self.is_connected or not self.ws:
            return False
        try:
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.commit",
                "event_id": f"commit_{time.time()}"
            }))
            return True
        except ConnectionClosed:
            print("Соединение закрыто при фиксации аудио")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"Ошибка фиксации аудио: {e}")
            return False

    async def clear_audio_buffer(self) -> bool:
        """Очищает аудио буфер, удаляя все ожидающие аудио данные"""
        if not self.is_connected or not self.ws:
            return False
        try:
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.clear",
                "event_id": f"clear_{time.time()}"
            }))
            return True
        except ConnectionClosed:
            print("Соединение закрыто при очистке аудио буфера")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"Ошибка очистки аудио буфера: {e}")
            return False
            
    async def cancel_response(self) -> bool:
        """Отменяет текущий ответ ассистента"""
        if not self.is_connected or not self.ws or not self.current_response_id:
            return False
        try:
            await self.ws.send(json.dumps({
                "type": "response.cancel",
                "event_id": f"cancel_{time.time()}",
                "response_id": self.current_response_id
            }))
            print(f"Отправлена команда отмены ответа: {self.current_response_id}")
            return True
        except ConnectionClosed:
            print("Соединение закрыто при отмене ответа")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"Ошибка отмены ответа: {e}")
            return False

    async def close(self) -> None:
        """Закрывает WebSocket соединение"""
        if self.ws:
            try:
                await self.ws.close()
                print(f"WebSocket соединение закрыто для клиента {self.client_id}")
            except Exception as e:
                print(f"Ошибка закрытия WebSocket OpenAI: {e}")
        self.is_connected = False
