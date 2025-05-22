import os
import asyncio
import json
import uuid
import base64
import time
import logging
import websockets
from typing import Optional, Dict, Any, Union
from websockets.exceptions import ConnectionClosed

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import uvicorn
import numpy as np

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# АУДИО УТИЛИТЫ
# =============================================================================

def base64_to_audio_buffer(base64_str: str) -> bytes:
    """Конвертация base64 строки в аудио буфер"""
    try:
        return base64.b64decode(base64_str)
    except Exception as e:
        logger.error(f"Ошибка конвертации base64 в аудио буфер: {e}")
        raise

def audio_buffer_to_base64(buffer: Union[bytes, bytearray, memoryview]) -> str:
    """Конвертация аудио буфера в base64 строку"""
    try:
        if isinstance(buffer, memoryview):
            buffer = buffer.tobytes()
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Ошибка конвертации аудио буфера в base64: {e}")
        raise

# =============================================================================
# OPENAI REALTIME CLIENT
# =============================================================================

class OpenAIRealtimeClient:
    """Клиент для взаимодействия с OpenAI Realtime API через WebSocket"""
    
    def __init__(self, api_key: str, client_id: str):
        self.api_key = api_key
        self.client_id = client_id
        self.ws: Optional[websockets.WebSocketServerProtocol] = None
        self.is_connected = False
        # ИСПРАВЛЕНИЕ: Модель указывается в URL, а не в session.update
        self.openai_url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        self.session_id = str(uuid.uuid4())
        
    async def connect(self) -> bool:
        """Установка WebSocket соединения с OpenAI Realtime API"""
        if not self.api_key:
            logger.error("OpenAI API ключ не предоставлен")
            return False
            
        headers = [
            ("Authorization", f"Bearer {self.api_key}"),
            ("OpenAI-Beta", "realtime=v1"),
            ("User-Agent", "VoiceAssistant/1.0")
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
            logger.info(f"Подключен к OpenAI для клиента {self.client_id}")
            
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
        """Переподключение к OpenAI после потери соединения"""
        logger.info(f"Попытка переподключения к OpenAI для клиента {self.client_id}")
        
        try:
            if self.ws:
                try:
                    await self.ws.close()
                except:
                    pass
            
            self.is_connected = False
            self.ws = None
            return await self.connect()
            
        except Exception as e:
            logger.error(f"Ошибка при переподключении к OpenAI: {e}")
            return False
    
    async def setup_session(self) -> bool:
        """Настройка сессии OpenAI с базовыми параметрами"""
        if not self.is_connected or not self.ws:
            logger.error("Нельзя настроить сессию: нет соединения")
            return False
        
        turn_detection = {
            "type": "server_vad",
            "threshold": 0.5,              # Увеличиваем порог для более надежного определения
            "prefix_padding_ms": 300,      # Больше отступа в начале
            "silence_duration_ms": 800,    # Больше времени ожидания тишины
            "create_response": True        # Автоматически создавать ответ
        }
        
        input_audio_transcription = {
            "model": "whisper-1"
        }
        
        # ИСПРАВЛЕНИЕ: Убираем поле "model" из session.update
        # Модель уже указана в URL подключения
        session_config = {
            "type": "session.update",
            "session": {
                "turn_detection": turn_detection,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "voice": "alloy",
                "instructions": "You are a helpful voice assistant. Respond briefly and naturally. Always respond when the user finishes speaking.",
                "modalities": ["text", "audio"],
                "temperature": 0.8,          # Увеличиваем для более естественных ответов
                "max_response_output_tokens": 150,  # Ограничиваем для быстрых ответов
                "input_audio_transcription": input_audio_transcription
            }
        }
        
        try:
            await self.ws.send(json.dumps(session_config))
            logger.info(f"Настройки сессии отправлены для клиента {self.client_id}")
            
            # Ждем подтверждение настройки сессии
            # OpenAI может отправить session.created, затем session.updated
            session_ready = False
            attempts = 0
            max_attempts = 3
            
            while not session_ready and attempts < max_attempts:
                try:
                    response = await asyncio.wait_for(self.ws.recv(), timeout=5)
                    response_data = json.loads(response)
                    msg_type = response_data.get("type")
                    
                    if msg_type == "session.created":
                        logger.info("Сессия создана OpenAI")
                        attempts += 1
                        continue  # Ждем session.updated
                        
                    elif msg_type == "session.updated":
                        logger.info("Сессия успешно обновлена")
                        session_ready = True
                        break
                        
                    elif msg_type == "error":
                        error_msg = response_data.get("error", {}).get("message", "Unknown error")
                        logger.error(f"Ошибка от OpenAI при настройке сессии: {error_msg}")
                        return False
                        
                    else:
                        logger.debug(f"Получено сообщение при настройке сессии: {msg_type}")
                        attempts += 1
                        
                except asyncio.TimeoutError:
                    logger.warning(f"Таймаут при ожидании подтверждения сессии (попытка {attempts + 1})")
                    attempts += 1
                    
            if session_ready:
                logger.info("Сессия готова к работе")
                return True
            else:
                logger.warning("Не получено подтверждение настройки сессии, но продолжаем работу")
                return True  # Все равно пробуем работать
                
        except Exception as e:
            logger.error(f"Ошибка настройки сессии: {e}")
            return False
    
    async def send_audio(self, audio_data: bytes) -> bool:
        """Отправка аудио данных в OpenAI"""
        if not self.is_connected or not self.ws or not audio_data:
            return False
            
        try:
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
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
        """Коммит аудио буфера"""
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
        """Очистка аудио буфера"""
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
        """Отмена текущего ответа от ассистента"""
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
    
    async def create_response(self) -> bool:
        """Принудительное создание ответа от ассистента"""
        if not self.is_connected or not self.ws:
            return False
            
        try:
            await self.ws.send(json.dumps({
                "type": "response.create",
                "event_id": f"response_{time.time()}"
            }))
            logger.info("Создан принудительный запрос ответа")
            return True
        except ConnectionClosed:
            logger.error("Соединение закрыто при создании ответа")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Ошибка создания ответа: {e}")
            return False
    
    async def close(self) -> None:
        """Закрытие WebSocket соединения"""
        if self.ws:
            try:
                await self.ws.close()
                logger.info(f"WebSocket соединение закрыто для клиента {self.client_id}")
            except Exception as e:
                logger.error(f"Ошибка закрытия OpenAI WebSocket: {e}")
        self.is_connected = False

# =============================================================================
# WEBSOCKET HANDLER
# =============================================================================

async def handle_websocket_connection(websocket: WebSocket):
    """Главный обработчик WebSocket соединения"""
    client_id = str(uuid.uuid4())
    openai_client = None
    
    try:
        await websocket.accept()
        logger.info(f"WebSocket соединение принято: client_id={client_id}")
        
        # Получаем API ключ из переменных окружения
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            await websocket.send_json({
                "type": "error",
                "error": {"code": "no_api_key", "message": "OpenAI API key not found in environment variables"}
            })
            await websocket.close(code=1008)
            return
        
        # Проверяем формат API ключа
        if not api_key.startswith("sk-"):
            await websocket.send_json({
                "type": "error",
                "error": {"code": "invalid_api_key", "message": "Invalid OpenAI API key format"}
            })
            await websocket.close(code=1008)
            return
        
        logger.info(f"Используем API ключ: {api_key[:10]}...{api_key[-4:]}")
        
        # Создаем и подключаем клиент OpenAI
        openai_client = OpenAIRealtimeClient(api_key, client_id)
        
        if not await openai_client.connect():
            await websocket.send_json({
                "type": "error", 
                "error": {"code": "openai_connection_failed", "message": "Failed to connect to OpenAI"}
            })
            await websocket.close(code=1008)
            return
        
        # Уведомляем клиента об успешном подключении
        await websocket.send_json({
            "type": "connection_status",
            "status": "connected", 
            "message": "Voice assistant ready"
        })
        
        # Буфер для накопления аудио данных
        audio_buffer = bytearray()
        
        # Запускаем задачу для обработки сообщений от OpenAI
        openai_task = asyncio.create_task(
            handle_openai_messages(openai_client, websocket)
        )
        
        # Основной цикл обработки сообщений от клиента
        while True:
            try:
                # Проверяем состояние соединения перед получением сообщения
                if websocket.client_state.name == 'DISCONNECTED':
                    logger.info(f"Клиент {client_id} отключен")
                    break
                    
                message = await websocket.receive()
                
                if "text" in message:
                    data = json.loads(message["text"])
                    msg_type = data.get("type", "")
                    
                    if msg_type == "ping":
                        await websocket.send_json({"type": "pong"})
                        
                    elif msg_type == "input_audio_buffer.append":
                        audio_chunk = base64_to_audio_buffer(data["audio"])
                        audio_buffer.extend(audio_chunk)
                        
                        if openai_client.is_connected:
                            await openai_client.send_audio(audio_chunk)
                        
                        await websocket.send_json({
                            "type": "input_audio_buffer.append.ack",
                            "event_id": data.get("event_id")
                        })
                        
                    elif msg_type == "input_audio_buffer.commit":
                        if len(audio_buffer) < 3200:  # ~100мс при 16kHz/16bit/mono
                            await websocket.send_json({
                                "type": "warning",
                                "warning": {
                                    "code": "audio_too_short",
                                    "message": "Audio too short, please speak longer"
                                }
                            })
                            audio_buffer.clear()
                            continue
                        
                        if openai_client.is_connected:
                            await openai_client.commit_audio()
                            
                            # ДОБАВЛЯЕМ: Принудительно создаем ответ если VAD не сработал
                            await asyncio.sleep(0.5)  # Небольшая пауза
                            await openai_client.create_response()
                            
                            await websocket.send_json({
                                "type": "input_audio_buffer.commit.ack",
                                "event_id": data.get("event_id")
                            })
                        
                        audio_buffer.clear()
                        
                    elif msg_type == "input_audio_buffer.clear":
                        audio_buffer.clear()
                        if openai_client.is_connected:
                            await openai_client.clear_audio_buffer()
                        await websocket.send_json({
                            "type": "input_audio_buffer.clear.ack",
                            "event_id": data.get("event_id")
                        })
                        
                    elif msg_type == "response.cancel":
                        if openai_client.is_connected:
                            await openai_client.cancel_response()
                        await websocket.send_json({
                            "type": "response.cancel.ack",
                            "event_id": data.get("event_id")
                        })
                        
                elif "bytes" in message:
                    audio_buffer.extend(message["bytes"])
                    if openai_client.is_connected:
                        await openai_client.send_audio(message["bytes"])
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket отключен: client_id={client_id}")
                break
            except ConnectionClosed:
                logger.info(f"Соединение закрыто: client_id={client_id}")
                break
            except RuntimeError as e:
                if "disconnect message has been received" in str(e):
                    logger.info(f"Клиент {client_id} уже отключился")
                    break
                else:
                    logger.error(f"Runtime ошибка в WebSocket цикле: {e}")
                    break
            except Exception as e:
                logger.error(f"Ошибка в WebSocket цикле: {e}")
                break
        
        # Отменяем задачу обработки OpenAI сообщений
        if not openai_task.done():
            openai_task.cancel()
            try:
                await openai_task
            except asyncio.CancelledError:
                pass
                
    except Exception as e:
        logger.error(f"Критическая ошибка в handle_websocket_connection: {e}")
        
    finally:
        if openai_client:
            await openai_client.close()
        logger.info(f"Соединение завершено: client_id={client_id}")

async def handle_openai_messages(openai_client: OpenAIRealtimeClient, websocket: WebSocket):
    """Обработчик сообщений от OpenAI Realtime API"""
    if not openai_client.is_connected or not openai_client.ws:
        logger.error("OpenAI клиент не подключен")
        return
    
    try:
        logger.info(f"Начало обработки сообщений от OpenAI для клиента {openai_client.client_id}")
        
        while True:
            try:
                raw_message = await openai_client.ws.recv()
                
                try:
                    response_data = json.loads(raw_message)
                except json.JSONDecodeError:
                    logger.error(f"Ошибка декодирования JSON от OpenAI: {raw_message[:200]}")
                    continue
                
                msg_type = response_data.get("type", "unknown")
                logger.info(f"Получено сообщение от OpenAI: тип={msg_type}")
                
                if msg_type == "error":
                    logger.error(f"Ошибка от OpenAI: {response_data}")
                    await websocket.send_json(response_data)
                    
                elif msg_type == "response.audio.delta":
                    audio_base64 = response_data.get("delta", "")
                    if audio_base64:
                        audio_chunk = base64.b64decode(audio_base64)
                        await websocket.send_bytes(audio_chunk)
                    
                elif msg_type == "conversation.item.input_audio_transcription.completed":
                    transcript = response_data.get("transcript", "")
                    logger.info(f"Пользователь сказал: '{transcript}'")
                    await websocket.send_json({
                        "type": "user_transcript",
                        "transcript": transcript
                    })
                    
                elif msg_type == "input_audio_buffer.speech_started":
                    logger.info("Обнаружено начало речи")
                    await websocket.send_json({
                        "type": "speech_started"
                    })
                    
                elif msg_type == "input_audio_buffer.speech_stopped":
                    logger.info("Обнаружено окончание речи")
                    await websocket.send_json({
                        "type": "speech_stopped"
                    })
                    
                elif msg_type == "response.created":
                    logger.info("OpenAI начал создавать ответ")
                    await websocket.send_json({
                        "type": "response_started"
                    })
                    
                elif msg_type == "response.audio_transcript.delta":
                    delta = response_data.get("delta", "")
                    if delta:
                        logger.info(f"Фрагмент ответа: '{delta}'")
                    
                elif msg_type == "response.audio_transcript.done":
                    transcript = response_data.get("transcript", "")
                    logger.info(f"Ассистент ответил: '{transcript}'")
                    await websocket.send_json({
                        "type": "assistant_transcript", 
                        "transcript": transcript
                    })
                    
                elif msg_type == "response.done":
                    logger.info("Ответ ассистента завершен")
                    await websocket.send_json({
                        "type": "response_completed"
                    })
                    
                else:
                    await websocket.send_json(response_data)
                    
            except ConnectionClosed:
                logger.warning("Соединение с OpenAI закрыто")
                if await openai_client.reconnect():
                    logger.info("Соединение с OpenAI восстановлено")
                    continue
                else:
                    logger.error("Не удалось восстановить соединение с OpenAI")
                    await websocket.send_json({
                        "type": "error",
                        "error": {
                            "code": "openai_connection_lost",
                            "message": "Connection to AI lost"
                        }
                    })
                    break
                    
    except (ConnectionClosed, asyncio.CancelledError):
        logger.info(f"Обработка сообщений OpenAI завершена для клиента {openai_client.client_id}")
    except Exception as e:
        logger.error(f"Ошибка в обработчике сообщений OpenAI: {e}")

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

# Создание FastAPI приложения
app = FastAPI(title="Voice Assistant", version="1.0.0")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Создание директории static если не существует
static_dir = "static"
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Подключение статических файлов
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Главная точка подключения WebSocket для голосового ассистента"""
    await handle_websocket_connection(websocket)

@app.get("/")
async def root():
    """Главная страница - отдаем HTML клиент"""
    try:
        return FileResponse('static/index.html')
    except:
        return {"message": "Voice Assistant API is running. Please create static/index.html file."}

@app.get("/health")
async def health_check():
    """Health check для Render"""
    return {"status": "healthy", "service": "voice-assistant"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False
    )
