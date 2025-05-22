import asyncio
import json
import uuid
import base64
import time
import websockets
import logging
from websockets.exceptions import ConnectionClosed
from typing import Optional

logger = logging.getLogger(__name__)

class OpenAIRealtimeClient:
    """
    Клиент для взаимодействия с OpenAI Realtime API через WebSocket.
    """
    
    def __init__(self, api_key: str, client_id: str):
        """
        Инициализация клиента OpenAI.
        
        Args:
            api_key: API ключ OpenAI
            client_id: Уникальный идентификатор клиента
        """
        self.api_key = api_key
        self.client_id = client_id
        self.ws: Optional[websockets.WebSocketServerProtocol] = None
        self.is_connected = False
        self.openai_url = "wss://api.openai.com/v1/realtime"
        self.session_id = str(uuid.uuid4())
        
    async def connect(self) -> bool:
        """
        Установка WebSocket соединения с OpenAI Realtime API.
        
        Returns:
            bool: True если соединение успешно, False иначе
        """
        if not self.api_key:
            logger.error("OpenAI API ключ не предоставлен")
            return False
            
        # Заголовки для аутентификации
        headers = [
            ("Authorization", f"Bearer {self.api_key}"),
            ("OpenAI-Beta", "realtime=v1"),
            ("User-Agent", "VoiceAssistant/1.0")
        ]
        
        try:
            # Устанавливаем WebSocket соединение
            self.ws = await asyncio.wait_for(
                websockets.connect(
                    self.openai_url,
                    extra_headers=headers,
                    max_size=15*1024*1024,  # 15 MB максимальный размер сообщения
                    ping_interval=30,       # Ping каждые 30 секунд
                    ping_timeout=120,       # Таймаут ping 120 секунд
                    close_timeout=15        # Таймаут закрытия 15 секунд
                ),
                timeout=30  # Таймаут подключения 30 секунд
            )
            
            self.is_connected = True
            logger.info(f"Подключен к OpenAI для клиента {self.client_id}")
            
            # Настраиваем сессию сразу после подключения
            if not await self.setup_session():
                logger.error("Не удалось настроить сессию OpenAI")
                await self.close()
                return False
                
            return True
            
        except asyncio.TimeoutError:
            logger.error(f"Таймаут подключения к OpenAI для клиента {self.client_id}")
            return False
        except Exception as e:
            logger.error(f"Ошибка подключения к OpenAI: {e}")
            return False
    
    async def reconnect(self) -> bool:
        """
        Переподключение к OpenAI после потери соединения.
        
        Returns:
            bool: True если переподключение успешно, False иначе
        """
        logger.info(f"Попытка переподключения к OpenAI для клиента {self.client_id}")
        
        try:
            # Закрываем старое соединение
            if self.ws:
                try:
                    await self.ws.close()
                except:
                    pass
            
            self.is_connected = False
            self.ws = None
            
            # Подключаемся заново
            return await self.connect()
            
        except Exception as e:
            logger.error(f"Ошибка при переподключении к OpenAI: {e}")
            return False
    
    async def setup_session(self) -> bool:
        """
        Настройка сессии OpenAI с базовыми параметрами.
        
        Returns:
            bool: True если настройка успешна, False иначе
        """
        if not self.is_connected or not self.ws:
            logger.error("Нельзя настроить сессию: нет соединения")
            return False
        
        # Настройки обнаружения речи (Voice Activity Detection)
        turn_detection = {
            "type": "server_vad",           # Серверное обнаружение активности голоса
            "threshold": 0.25,              # Порог обнаружения речи (0.0-1.0)
            "prefix_padding_ms": 200,       # Отступ в начале речи (мс)
            "silence_duration_ms": 300,     # Длительность молчания для завершения (мс)
            "create_response": True         # Автоматически создавать ответ
        }
        
        # Настройки транскрипции входящего аудио
        input_audio_transcription = {
            "model": "whisper-1"
        }
        
        # Конфигурация сессии
        session_config = {
            "type": "session.update",
            "session": {
                "turn_detection": turn_detection,
                "input_audio_format": "pcm16",      # Формат входящего аудио
                "output_audio_format": "pcm16",     # Формат исходящего аудио
                "voice": "alloy",                   # Голос для синтеза речи
                "instructions": "You are a helpful voice assistant. Respond briefly and naturally.",
                "modalities": ["text", "audio"],    # Поддерживаемые модальности
                "temperature": 0.7,                 # Креативность ответов
                "max_response_output_tokens": 500,  # Максимум токенов в ответе
                "input_audio_transcription": input_audio_transcription
            }
        }
        
        try:
            # Отправляем конфигурацию сессии
            await self.ws.send(json.dumps(session_config))
            logger.info(f"Сессия настроена для клиента {self.client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка настройки сессии: {e}")
            return False
    
    async def send_audio(self, audio_data: bytes) -> bool:
        """
        Отправка аудио данных в OpenAI.
        
        Args:
            audio_data: Бинарные аудио данные в формате PCM16
            
        Returns:
            bool: True если успешно, False иначе
        """
        if not self.is_connected or not self.ws or not audio_data:
            return False
            
        try:
            # Кодируем аудио в base64
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            
            # Отправляем в OpenAI
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": audio_base64,
                "event_id": f"audio_{time.time()}"
            }))
            
            return True
            
        except ConnectionClosed:
            logger.error("Соединение закрыто при отправке аудио")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Ошибка отправки аудио: {e}")
            return False
    
    async def commit_audio(self) -> bool:
        """
        Коммит аудио буфера (сигнал о завершении речи пользователя).
        
        Returns:
            bool: True если успешно, False иначе
        """
        if not self.is_connected or not self.ws:
            return False
            
        try:
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.commit",
                "event_id": f"commit_{time.time()}"
            }))
            return True
            
        except ConnectionClosed:
            logger.error("Соединение закрыто при коммите аудио")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Ошибка коммита аудио: {e}")
            return False
    
    async def clear_audio_buffer(self) -> bool:
        """
        Очистка аудио буфера.
        
        Returns:
            bool: True если успешно, False иначе
        """
        if not self.is_connected or not self.ws:
            return False
            
        try:
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.clear",
                "event_id": f"clear_{time.time()}"
            }))
            return True
            
        except ConnectionClosed:
            logger.error("Соединение закрыто при очистке буфера")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Ошибка очистки буфера: {e}")
            return False
    
    async def cancel_response(self) -> bool:
        """
        Отмена текущего ответа от ассистента.
        
        Returns:
            bool: True если успешно, False иначе
        """
        if not self.is_connected or not self.ws:
            return False
            
        try:
            await self.ws.send(json.dumps({
                "type": "response.cancel",
                "event_id": f"cancel_{time.time()}"
            }))
            return True
            
        except ConnectionClosed:
            logger.error("Соединение закрыто при отмене ответа")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Ошибка отмены ответа: {e}")
            return False
    
    async def close(self) -> None:
        """
        Закрытие WebSocket соединения.
        """
        if self.ws:
            try:
                await self.ws.close()
                logger.info(f"WebSocket соединение закрыто для клиента {self.client_id}")
            except Exception as e:
                logger.error(f"Ошибка закрытия OpenAI WebSocket: {e}")
        
        self.is_connected = False
