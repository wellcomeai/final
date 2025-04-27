import os
import json
import base64
import asyncio
import logging
import traceback
import shutil
import websockets
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import Dict, Optional, List, Any
from pydantic import BaseModel
import time
import uuid
from fastapi import Body, Header, Depends, HTTPException, Request, UploadFile, File

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wellcome-ai")
logger.setLevel(logging.DEBUG)

# Загружаем переменные окружения
load_dotenv()

# Базовое содержимое HTML для заглушки
DEFAULT_HTML_CONTENT = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>WellcomeAI</title>
  <style>
    body { 
      font-family: 'Segoe UI', sans-serif; 
      background: white; 
      display: flex; 
      justify-content: center;
      align-items: center;
      height: 100vh;
      margin: 0;
    }
    .container { 
      text-align: center; 
      max-width: 600px;
      padding: 20px;
    }
    h1 { color: #4a86e8; }
  </style>
</head>
<body>
  <div class="container">
    <h1>WellcomeAI</h1>
    <p>Загрузка...</p>
  </div>
</body>
</html>
"""

# Конфигурация
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PORT = int(os.getenv('PORT', 5050))
REALTIME_WS_URL = 'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01'
SYSTEM_MESSAGE = (
    "Ты WellcomeAI - умный голосовой помощник. Отвечай на вопросы пользователя коротко, "
    "информативно и с небольшой ноткой юмора, когда это уместно. Стремись быть полезным "
    "и предоставлять точную информацию. Избегай длинных вступлений и лишних фраз."
)

# Допустимые голоса с русскими названиями для интерфейса
AVAILABLE_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]
VOICE_NAMES = {
    "alloy": "Alloy",
    "ash": "Ash",
    "ballad": "Ballad",
    "coral": "Coral",
    "echo": "Echo",
    "sage": "Sage",
    "shimmer": "Shimmer",
    "verse": "Verse"
}
DEFAULT_VOICE = "alloy"

# Проверяем наличие директории static и создаем ее, если она не существует
static_dir = os.path.join(os.getcwd(), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
    logger.info(f"Создана директория static")

# Проверяем наличие index.html в корне и копируем в static
index_in_root = os.path.join(os.getcwd(), "index.html")
index_in_static = os.path.join(static_dir, "index.html")

# Если файл есть в корне, копируем его в static
if os.path.exists(index_in_root):
    try:
        shutil.copy2(index_in_root, index_in_static)
        logger.info("index.html скопирован из корня в static директорию")
    except Exception as e:
        logger.error(f"Ошибка при копировании index.html: {str(e)}")

# Если файла нет в static, создаем заглушку
if not os.path.exists(index_in_static):
    try:
        with open(index_in_static, "w", encoding="utf-8") as f:
            f.write(DEFAULT_HTML_CONTENT)
        logger.info("Создан файл index.html в директории static")
    except Exception as e:
        logger.error(f"Ошибка при создании index.html: {str(e)}")

app = FastAPI()

# Добавляем CORS middleware для разрешения кросс-доменных запросов
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

if not OPENAI_API_KEY:
    logger.warning('Отсутствует ключ API OpenAI. Пожалуйста, укажите его в .env файле.')

# Отслеживаемые события от OpenAI для подробного логирования
LOG_EVENT_TYPES = [
    'response.done',
    'input_audio_buffer.speech_stopped',
    'input_audio_buffer.speech_started',
    'session.created', 
    'session.updated'
]

# Хранилище активных соединений клиент <-> OpenAI
client_connections = {}

# Модели данных для API
class SessionRequest(BaseModel):
    prompt_id: Optional[str] = None
    record_id: Optional[str] = None
    lang: Optional[str] = "ru"
    voice: Optional[bool] = True
    avatar: Optional[bool] = False
    wait_msg: Optional[str] = "Думаю..."

class TextRequest(BaseModel):
    session_id: str
    message: str

class VoiceSettings(BaseModel):
    voice_id: str = DEFAULT_VOICE

class ApiSession(BaseModel):
    session_id: str
    created_at: float
    last_active: float
    openai_session_id: Optional[str] = None
    voice: str = DEFAULT_VOICE
    lang: str = "ru"
    messages: List[Dict[str, Any]] = []

# Хранилище сессий API (в реальном приложении используйте базу данных)
api_sessions: Dict[str, ApiSession] = {}

# Функция очистки старых сессий
async def cleanup_sessions():
    while True:
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in api_sessions.items():
            # Удаляем сессии старше 30 минут бездействия
            if current_time - session.last_active > 1800:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            logger.info(f"Удаление неактивной сессии API: {session_id}")
            del api_sessions[session_id]
            
            # Также закрываем соединение с OpenAI, если оно существует
            for client_id, client_data in list(client_connections.items()):
                if hasattr(client_data, 'api_session_id') and client_data.api_session_id == session_id:
                    # Закрываем соединение
                    if client_data["openai_ws"]:
                        try:
                            await client_data["openai_ws"].close()
                        except:
                            pass
                    
                    # Удаляем информацию о клиенте
                    client_connections[client_id]["active"] = False
                    del client_connections[client_id]
        
        await asyncio.sleep(300)  # Проверка каждые 5 минут

# Запускаем задачу очистки при старте приложения
@app.on_event("startup")
async def start_cleanup_task():
    asyncio.create_task(cleanup_sessions())

@app.get("/")
async def index_page():
    """Возвращает HTML страницу с интерфейсом голосового помощника"""
    try:
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as file:
                content = file.read()
            return HTMLResponse(content=content)
        else:
            # Если файл не найден в static, создаем заглушку
            logger.warning(f"Файл {index_path} не найден, создаем заглушку")
            with open(index_path, "w", encoding="utf-8") as file:
                file.write(DEFAULT_HTML_CONTENT)
            with open(index_path, "r", encoding="utf-8") as file:
                content = file.read()
            return HTMLResponse(content=content)
    except Exception as e:
        logger.error(f"Ошибка при отдаче главной страницы: {str(e)}")
        return HTMLResponse(
            content=f"<html><body><h1>WellcomeAI</h1><p>Произошла ошибка: {str(e)}</p></body></html>",
            status_code=500
        )

@app.get("/api/check")
async def check_api():
    """Проверка состояния API и доступности OpenAI"""
    try:
        # Проверяем, что ключ API существует
        if not OPENAI_API_KEY:
            return {
                "status": "error",
                "message": "API ключ не настроен на сервере"
            }
            
        # Возвращаем успешный статус
        return {
            "status": "success",
            "api_key_valid": True,
            "openai_api_version": "Realtime API",
            "available_voices": AVAILABLE_VOICES,
            "voice_names": VOICE_NAMES
        }
    except Exception as e:
        logger.error(f"Ошибка при проверке API: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

async def create_openai_connection():
    """Создание нового соединения с OpenAI API"""
    try:
        openai_ws = await websockets.connect(
            REALTIME_WS_URL,
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            },
            # Увеличены буферы для более надежной передачи аудио
            max_size=10 * 1024 * 1024,  # 10MB max message size
            ping_interval=20,
            ping_timeout=60
        )
        logger.info("Создано новое соединение с OpenAI")
        return openai_ws
    except Exception as e:
        logger.error(f"Ошибка при создании соединения с OpenAI: {str(e)}")
        raise

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Основной WebSocket-эндпоинт для голосового взаимодействия
    Устанавливает два соединения: с клиентом и с OpenAI Realtime API
    """
    client_id = id(websocket)
    logger.info(f"Новое клиентское соединение: {client_id}")
    
    await websocket.accept()
    
    # Хранение информации об этом клиенте
    client_connections[client_id] = {
        "client_ws": websocket,
        "openai_ws": None,
        "active": True,
        "voice": DEFAULT_VOICE,
        "turn_detection_enabled": True,
        "tasks": [],     # Для хранения задач
        "reconnecting": False  # Флаг, указывающий на пересоздание соединения
    }
    
    try:
        # Устанавливаем соединение с OpenAI
        openai_ws = await asyncio.wait_for(
            create_openai_connection(),
            timeout=20.0
        )
        
        client_connections[client_id]["openai_ws"] = openai_ws
        logger.info(f"Соединение с OpenAI установлено для клиента {client_id}")
        
        # Отправляем настройки сессии в OpenAI
        await send_session_update(openai_ws, DEFAULT_VOICE)
        
        # Создаем две задачи для двустороннего обмена сообщениями
        client_to_openai = asyncio.create_task(forward_client_to_openai(websocket, openai_ws, client_id))
        openai_to_client = asyncio.create_task(forward_openai_to_client(openai_ws, websocket, client_id))
        
        # Сохраняем задачи для возможности отмены
        client_connections[client_id]["tasks"] = [client_to_openai, openai_to_client]
        
        # Ждем, пока одна из задач не завершится
        done, pending = await asyncio.wait(
            [client_to_openai, openai_to_client],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Проверяем, есть ли ошибка в завершенных задачах
        for task in done:
            try:
                # Если задача завершилась с ошибкой, вызываем исключение
                task.result()
            except Exception as e:
                logger.error(f"Задача завершилась с ошибкой: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Отменяем оставшиеся задачи
        for task in pending:
            task.cancel()
        
    except Exception as e:
        logger.error(f"Ошибка при обработке WebSocket соединения: {str(e)}")
        logger.error(traceback.format_exc())
        try:
            await websocket.send_json({
                "type": "error",
                "error": {
                    "message": f"Произошла ошибка: {str(e)}"
                }
            })
        except:
            pass
    finally:
        # Закрываем соединение с OpenAI, если оно существует
        if client_id in client_connections and client_connections[client_id]["openai_ws"]:
            try:
                await client_connections[client_id]["openai_ws"].close()
            except:
                pass
        
        # Отменяем все задачи
        if client_id in client_connections and "tasks" in client_connections[client_id]:
            for task in client_connections[client_id]["tasks"]:
                if not task.done():
                    task.cancel()
        
        # Удаляем информацию о клиенте
        if client_id in client_connections:
            client_connections[client_id]["active"] = False
            del client_connections[client_id]
        
        logger.info(f"Соединение с клиентом {client_id} закрыто")

async def recreate_openai_connection(client_id, new_voice):
    """Пересоздание соединения с OpenAI для смены голоса"""
    if client_id not in client_connections:
        logger.error(f"Клиент {client_id} не найден при попытке пересоздания соединения")
        return False
    
    try:
        client_data = client_connections[client_id]
        client_ws = client_data["client_ws"]
        
        # Устанавливаем флаг переподключения
        client_data["reconnecting"] = True
        
        # Уведомляем клиента о пересоздании соединения
        try:
            await client_ws.send_json({
                "type": "recreating_connection",
                "message": "Переключение голоса..."
            })
        except Exception as e:
            logger.error(f"Ошибка при отправке уведомления о пересоздании: {str(e)}")
        
        # Закрываем текущее соединение
        if client_data["openai_ws"]:
            try:
                await client_data["openai_ws"].close()
            except Exception as e:
                logger.error(f"Ошибка при закрытии старого соединения: {str(e)}")
        
        # Отменяем текущие задачи
        for task in client_data.get("tasks", []):
            if not task.done():
                task.cancel()
                
        # Небольшая пауза для завершения задач
        await asyncio.sleep(1.0)  # Увеличил паузу для надежности
        
        # Создаем новое соединение
        new_openai_ws = await create_openai_connection()
        client_connections[client_id]["openai_ws"] = new_openai_ws
        
        # Сохраняем новый голос в данных клиента перед отправкой настроек
        client_connections[client_id]["voice"] = new_voice
        
        # Отправляем настройки сессии с новым голосом
        await send_session_update(new_openai_ws, new_voice)
        
        # Дополнительная пауза, чтобы убедиться, что настройки применились
        await asyncio.sleep(0.5)
        
        # Создаем новые задачи
        client_to_openai = asyncio.create_task(forward_client_to_openai(client_ws, new_openai_ws, client_id))
        openai_to_client = asyncio.create_task(forward_openai_to_client(new_openai_ws, client_ws, client_id))
        
        # Сохраняем новые задачи
        client_connections[client_id]["tasks"] = [client_to_openai, openai_to_client]
        
        # Небольшая пауза, чтобы задачи запустились
        await asyncio.sleep(0.5)
        
        # Сбрасываем флаг переподключения
        client_connections[client_id]["reconnecting"] = False
        
        # Уведомляем клиента об успешной смене голоса
        voice_name = VOICE_NAMES.get(new_voice, new_voice)
        await client_ws.send_json({
            "type": "voice_changed",
            "voice": new_voice,
            "voice_name": voice_name,
            "success": True,
            "ready": True  # Флаг готовности к работе
        })
        
        logger.info(f"Голос для клиента {client_id} успешно изменен на {new_voice}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при пересоздании соединения: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Сбрасываем флаг переподключения
        if client_id in client_connections:
            client_connections[client_id]["reconnecting"] = False
        
        # Уведомляем клиента об ошибке
        try:
            if client_id in client_connections and client_connections[client_id]["client_ws"]:
                await client_connections[client_id]["client_ws"].send_json({
                    "type": "voice_changed",
                    "voice": DEFAULT_VOICE,
                    "voice_name": VOICE_NAMES.get(DEFAULT_VOICE, DEFAULT_VOICE),
                    "message": "Ошибка при смене голоса",
                    "success": False
                })
        except:
            pass
        
        return False

async def forward_client_to_openai(client_ws: WebSocket, openai_ws, client_id: int):
    """Пересылает сообщения от клиента (браузера) к API OpenAI"""
    try:
        while client_id in client_connections and client_connections[client_id]["active"]:
            # Получаем данные от клиента
            try:
                message = await client_ws.receive_text()
            except WebSocketDisconnect:
                logger.info(f"Клиент {client_id} отключился")
                break
            
            # Проверяем, что сообщение не пустое
            if not message:
                continue
                
            # Проверяем, что клиент не в процессе переподключения
            if client_id in client_connections and client_connections[client_id].get("reconnecting", False):
                logger.debug(f"Сообщение от клиента {client_id} проигнорировано - идет переподключение")
                continue
            
            # Парсим JSON
            try:
                data = json.loads(message)
                msg_type = data.get("type", "unknown")
                
                # Команда смены голоса
                if msg_type == "change_voice":
                    new_voice = data.get("voice", DEFAULT_VOICE)
                    if new_voice in AVAILABLE_VOICES:
                        logger.info(f"Запрос на смену голоса на {new_voice}")
                        # Пересоздаем соединение для новой сессии с новым голосом
                        await recreate_openai_connection(client_id, new_voice)
                        continue
                    else:
                        logger.warning(f"Запрошенный голос {new_voice} недоступен")
                        await client_ws.send_json({
                            "type": "voice_changed",
                            "voice": client_connections[client_id]["voice"],
                            "voice_name": VOICE_NAMES.get(client_connections[client_id]["voice"], client_connections[client_id]["voice"]),
                            "message": f"Голос {new_voice} недоступен",
                            "success": False
                        })
                        continue
                
                # Если это обновление сессии, обрабатываем его
                if msg_type == "session.update":
                    # Обновляем настройки сессии
                    session_data = data.get("session", {})
                    
                    # Если указан голос, пересоздаем соединение вместо обновления
                    if "voice" in session_data:
                        new_voice = session_data["voice"]
                        if new_voice in AVAILABLE_VOICES and new_voice != client_connections[client_id]["voice"]:
                            logger.info(f"Запрос на смену голоса через session.update на {new_voice}")
                            # Пересоздаем соединение с новым голосом
                            await recreate_openai_connection(client_id, new_voice)
                            continue
                
                # Не логируем аппенд аудио буфера для уменьшения шума в логах
                if msg_type != "input_audio_buffer.append":
                    logger.debug(f"[Клиент {client_id} -> OpenAI] {msg_type}")
                
                # Отправляем сообщение в OpenAI
                await openai_ws.send(message)
                
            except json.JSONDecodeError as e:
                logger.error(f"Получены некорректные данные от клиента")
            except Exception as e:
                logger.error(f"Ошибка при обработке сообщения от клиента: {str(e)}")
                logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"Ошибка в задаче forward_client_to_openai: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def forward_openai_to_client(openai_ws, client_ws: WebSocket, client_id: int):
    """Пересылает сообщения от API OpenAI клиенту (браузеру)"""
    try:
        async for openai_message in openai_ws:
            if client_id not in client_connections or not client_connections[client_id]["active"]:
                break
                
            try:
                # Парсим JSON от OpenAI
                if isinstance(openai_message, str):
                    response = json.loads(openai_message)
                    
                    # Логируем определенные типы событий
                    if response.get('type') in LOG_EVENT_TYPES:
                        logger.info(f"[OpenAI -> Клиент {client_id}] {response.get('type')}")
                        
                        # Если получено событие создания сессии после смены голоса,
                        # отправляем уведомление о готовности
                        if response.get('type') == 'session.created' and client_connections[client_id].get("reconnecting", False):
                            await client_ws.send_json({
                                "type": "session_ready",
                                "voice": client_connections[client_id]["voice"],
                                "voice_name": VOICE_NAMES.get(client_connections[client_id]["voice"], client_connections[client_id]["voice"])
                            })
                    
                    # Обрабатываем ошибки сервера
                    if response.get('type') == 'error':
                        error_msg = response.get('error', {}).get('message', 'Неизвестная ошибка')
                        logger.error(f"Ошибка от OpenAI: {error_msg}")
                        
                        # Если это ошибка, связанная с невозможностью изменить голос
                        if 'Cannot update a conversation\'s voice' in error_msg:
                            # Это значит, что нужно пересоздать соединение для смены голоса
                            logger.info("Обнаружена необходимость пересоздания соединения для смены голоса")
                            
                            # Пересоздаем соединение с текущим голосом
                            current_voice = client_connections[client_id]["voice"]
                            await recreate_openai_connection(client_id, current_voice)
                            
                            # Продолжаем без отправки этой ошибки клиенту
                            continue
                        
                        # Если это ошибка пустого буфера, не показываем пользователю
                        if 'buffer too small' in error_msg or 'Expected at least 100ms' in error_msg:
                            # Отправляем специальное сообщение для диагностики
                            await client_ws.send_json({
                                "type": "buffer_error",
                                "message": "Недостаточно аудио для обработки. Говорите немного дольше."
                            })
                            continue
                    
                    # Пересылаем сообщение клиенту
                    await client_ws.send_text(openai_message)
                else:
                    # Если это бинарные данные, отправляем как есть
                    await client_ws.send_bytes(openai_message)
                
            except json.JSONDecodeError:
                logger.error(f"Получены некорректные данные от OpenAI")
                # Пытаемся все равно отправить данные клиенту
                if isinstance(openai_message, str):
                    await client_ws.send_text(openai_message)
                else:
                    await client_ws.send_bytes(openai_message)
            except Exception as e:
                logger.error(f"Ошибка при пересылке данных от OpenAI клиенту: {str(e)}")
    
    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Соединение с OpenAI закрыто для клиента {client_id}")
        try:
            # Сообщаем клиенту о закрытии соединения
            await client_ws.send_json({
                "type": "error",
                "error": {
                    "message": "Соединение прервано"
                }
            })
        except:
            pass
    except Exception as e:
        logger.error(f"Ошибка в задаче forward_openai_to_client: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def send_session_update(openai_ws, voice=DEFAULT_VOICE, turn_detection_enabled=True):
    """Отправляет настройки сессии в WebSocket OpenAI"""
    
    # Настройка определения завершения речи
    turn_detection = {
        "type": "server_vad",
        "threshold": 0.25,                 # Чувствительность определения голоса
        "prefix_padding_ms": 200,          # Начальное время записи
        "silence_duration_ms": 300,        # Время ожидания тишины
        "create_response": True            # Автоматически создавать ответ при завершении речи
    } if turn_detection_enabled else None
    
    # Логируем отправляемый голос для диагностики
    logger.info(f"Подготовка настроек сессии с голосом: {voice}")
    
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": turn_detection,
            "input_audio_format": "pcm16",        # Формат входящего аудио
            "output_audio_format": "pcm16",       # Формат исходящего аудио
            "voice": voice,                       # Голос ассистента
            "instructions": SYSTEM_MESSAGE,       # Системное сообщение
            "modalities": ["text", "audio"],      # Поддерживаемые модальности
            "temperature": 0.7,                   # Температура генерации
            "max_response_output_tokens": 500     # Лимит токенов для ответа
        }
    }
    
    # Дамп настроек для диагностики проблем с голосом
    logger.debug(f"Настройки сессии: {json.dumps(session_update)}")
    
    try:
        # Отправляем настройки и ожидаем небольшое время для применения
        await openai_ws.send(json.dumps(session_update))
        logger.info(f"Настройки сессии с голосом {voice} отправлены")
    except Exception as e:
        logger.error(f"Ошибка при отправке настроек сессии: {str(e)}")
        raise

# Функция для создания WebSocket соединения для REST API сессии
async def create_api_websocket_connection(session_id: str, voice_id: str = DEFAULT_VOICE):
    """
    Создает WebSocket соединение с OpenAI для сессии API
    и настраивает его для прослушивания и ответов.
    """
    try:
        # Уникальный ID для клиента
        client_id = f"api_{str(uuid.uuid4())}"
        
        # Создаем WebSocket соединение с OpenAI
        openai_ws = await create_openai_connection()
        
        # Сохраняем информацию о соединении
        client_connections[client_id] = {
            "client_ws": None,  # В REST API нет клиентского WebSocket
            "openai_ws": openai_ws,
            "active": True,
            "voice": voice_id,
            "api_session_id": session_id,  # Связываем с API сессией
            "tasks": [],
            "reconnecting": False,
            "messages_queue": asyncio.Queue(),  # Очередь для сообщений от OpenAI
            "response_ready": asyncio.Event()   # Событие для сигнализации о готовности ответа
        }
        
        # Отправляем настройки сессии в OpenAI
        await send_session_update(openai_ws, voice_id)
        
        # Создаем задачу для слушания сообщений от OpenAI
        openai_listener = asyncio.create_task(
            listen_openai_for_api(openai_ws, client_id, session_id)
        )
        
        # Сохраняем задачу
        client_connections[client_id]["tasks"].append(openai_listener)
        
        # Обновляем сессию API с ID клиента
        if session_id in api_sessions:
            api_sessions[session_id].openai_session_id = client_id
        
        logger.info(f"Создано API соединение с OpenAI для сессии {session_id}, клиент {client_id}")
        return client_id
    
    except Exception as e:
        logger.error(f"Ошибка при создании API соединения: {str(e)}")
        logger.error(traceback.format_exc())
        return None

async def listen_openai_for_api(openai_ws, client_id: str, session_id: str):
    """
    Слушает сообщения от OpenAI для API сессии и складывает их в очередь
    """
    try:
        text_response = ""
        audio_chunks = []
        
        async for openai_message in openai_ws:
            if client_id not in client_connections or not client_connections[client_id]["active"]:
                break
                
            try:
                # Парсим JSON от OpenAI
                if isinstance(openai_message, str):
                    response = json.loads(openai_message)
                    
                    # Логируем определенные типы событий
                    if response.get('type') in LOG_EVENT_TYPES:
                        logger.info(f"[OpenAI -> API клиент {client_id}] {response.get('type')}")
                    
                    # Обрабатываем текстовый ответ
                    if response.get('type') == 'response.text.delta':
                        if 'delta' in response:
                            text_response += response['delta']
                    
                    # Обрабатываем завершение ответа
                    if response.get('type') == 'response.done':
                        # Сохраняем результат в очередь
                        await client_connections[client_id]["messages_queue"].put({
                            "type": "response_complete",
                            "text": text_response,
                            "audio": audio_chunks,
                            "timestamp": time.time()
                        })
                        
                        # Сигнализируем о готовности ответа
                        client_connections[client_id]["response_ready"].set()
                        
                        # Сбрасываем для следующего ответа
                        text_response = ""
                        audio_chunks = []
                    
                    # Обрабатываем ошибки
                    if response.get('type') == 'error':
                        error_msg = response.get('error', {}).get('message', 'Неизвестная ошибка')
                        await client_connections[client_id]["messages_queue"].put({
                            "type": "error",
                            "message": error_msg,
                            "timestamp": time.time()
                        })
                        client_connections[client_id]["response_ready"].set()
                
                # Обрабатываем аудио чанки
                if response.get('type') == 'response.audio.delta':
                    if 'delta' in response:
                        audio_chunks.append(response['delta'])
                
            except json.JSONDecodeError:
                logger.error(f"Получены некорректные данные от OpenAI для API клиента {client_id}")
            except Exception as e:
                logger.error(f"Ошибка при обработке данных от OpenAI для API клиента {client_id}: {str(e)}")
    
    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Соединение с OpenAI закрыто для API клиента {client_id}")
    except Exception as e:
        logger.error(f"Ошибка в задаче listen_openai_for_api: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Помечаем соединение как неактивное при выходе
        if client_id in client_connections:
            client_connections[client_id]["active"] = False

# API эндпоинты для REST API
@app.get("/widget")
async def widget_page():
    """Возвращает HTML страницу с интерфейсом голосового помощника для встраивания"""
    widget_path = os.path.join(static_dir, "widget.html")
    
    # Если файл виджета не существует, используем стандартный index.html
    if not os.path.exists(widget_path):
        widget_path = os.path.join(static_dir, "index.html")
    
    try:
        with open(widget_path, "r", encoding="utf-8") as file:
            content = file.read()
        return HTMLResponse(content=content)
    except Exception as e:
        logger.error(f"Ошибка при отдаче виджета: {str(e)}")
        return HTMLResponse(
            content=f"<html><body><h1>Ошибка</h1><p>{str(e)}</p></body></html>",
            status_code=500
        )

# Создаем новую сессию для API
@app.post("/api/v1/session")
async def create_session(request: SessionRequest = Body(...)):
    session_id = str(uuid.uuid4())
    
    # Создаем новую сессию
    session = ApiSession(
        session_id=session_id,
        created_at=time.time(),
        last_active=time.time(),
        voice=DEFAULT_VOICE,
        lang=request.lang or "ru",
        messages=[]
    )
    
    api_sessions[session_id] = session
    
    return {
        "status": "success",
        "session_id": session_id,
        "message": "Сессия создана успешно"
    }

# Получаем данные сессии
@app.get("/api/v1/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in api_sessions:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    
    session = api_sessions[session_id]
    session.last_active = time.time()
    
    return {
        "status": "success",
        "session": {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "voice": session.voice,
            "lang": session.lang,
            "message_count": len(session.messages)
        }
    }

# Изменяем настройки сессии
@app.put("/api/v1/session/{session_id}/voice")
async def update_session_voice(session_id: str, settings: VoiceSettings):
    if session_id not in api_sessions:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    
    session = api_sessions[session_id]
    session.last_active = time.time()
    
    voice_id = settings.voice_id
    if voice_id not in AVAILABLE_VOICES:
        raise HTTPException(status_code=400, detail=f"Голос {voice_id} недоступен")
    
    # Изменяем голос в сессии
    session.voice = voice_id
    
    # Если есть активное соединение с OpenAI для этой сессии, изменяем и его
    for client_id, client_data in client_connections.items():
        if hasattr(client_data, 'api_session_id') and client_data.api_session_id == session_id:
            # Запускаем пересоздание соединения для смены голоса
            await recreate_openai_connection(client_id, voice_id)
    
    return {
        "status": "success",
        "message": f"Голос изменен на {voice_id}"
    }

# Отправка текстового сообщения
@app.post("/api/v1/message")
async def send_text_message(request: TextRequest):
    session_id = request.session_id
    
    if session_id not in api_sessions:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    
    session = api_sessions[session_id]
    session.last_active = time.time()
    
    # Добавляем сообщение пользователя в историю
    user_message = {
        "role": "user",
        "content": request.message,
        "timestamp": time.time()
    }
    session.messages.append(user_message)
    
    # Проверяем, есть ли активное соединение с OpenAI для этой сессии
    client_id = None
    if session.openai_session_id and session.openai_session_id in client_connections:
        client_id = session.openai_session_id
    
    # Если нет активного соединения, создаем новое
    if client_id is None:
        client_id = await create_api_websocket_connection(session_id, session.voice)
        if not client_id:
            raise HTTPException(status_code=500, detail="Не удалось создать соединение с OpenAI")
    
    # Отправляем сообщение через WebSocket
    try:
        # Сбрасываем событие готовности ответа
        client_connections[client_id]["response_ready"].clear()
        
        # Отправляем запрос в OpenAI
        conversation_item = {
            "type": "conversation.item.create",
            "event_id": f"msg_{time.time()}",
            "item": {
                "id": f"user_msg_{uuid.uuid4()}",
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": request.message
                    }
                ]
            }
        }
        
        await client_connections[client_id]["openai_ws"].send(json.dumps(conversation_item))
        
        # Отправляем команду для создания ответа
        response_create = {
            "type": "response.create",
            "event_id": f"resp_{time.time()}",
            "response": {
                "temperature": 0.7
            }
        }
        
        await client_connections[client_id]["openai_ws"].send(json.dumps(response_create))
        
        # Ждем ответа с таймаутом
        try:
            await asyncio.wait_for(
                client_connections[client_id]["response_ready"].wait(),
                timeout=60.0  # Максимальное время ожидания - 60 секунд
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Превышено время ожидания ответа")
        
        # Получаем ответ из очереди
        response_data = await client_connections[client_id]["messages_queue"].get()
        
        if response_data["type"] == "error":
            raise HTTPException(status_code=500, detail=response_data["message"])
        
        # Получаем текст и аудио
        response_text = response_data["text"]
        audio_chunks = response_data["audio"]
        
        # Объединяем аудио чанки (если есть)
        response_audio = None
        if audio_chunks:
            response_audio = "".join(audio_chunks)
        
        # Добавляем ответ ассистента в историю
        assistant_message = {
            "role": "assistant",
            "content": response_text,
            "timestamp": time.time()
        }
        session.messages.append(assistant_message)
        
        return {
            "status": "success",
            "response": {
                "text": response_text,
                "audio": response_audio
            }
        }
    
    except Exception as e:
        logger.error(f"Ошибка при обработке текстового запроса: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

# Загрузка и обработка аудиофайла
@app.post("/api/v1/audio")
async def process_audio(
    session_id: str = Body(...),
    audio_file: UploadFile = File(...)
):
    if session_id not in api_sessions:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    
    session = api_sessions[session_id]
    session.last_active = time.time()
    
    # Проверяем, есть ли активное соединение с OpenAI для этой сессии
    client_id = None
    if session.openai_session_id and session.openai_session_id in client_connections:
        client_id = session.openai_session_id
    
    # Если нет активного соединения, создаем новое
    if client_id is None:
        client_id = await create_api_websocket_connection(session_id, session.voice)
        if not client_id:
            raise HTTPException(status_code=500, detail="Не удалось создать соединение с OpenAI")
    
    try:
        # Сбрасываем событие готовности ответа
        client_connections[client_id]["response_ready"].clear()
        
        # Читаем аудиофайл
        audio_content = await audio_file.read()
        
        # Отправляем аудио через WebSocket
        await client_connections[client_id]["openai_ws"].send(json.dumps({
            "type": "input_audio_buffer.clear",
            "event_id": f"clear_{time.time()}"
        }))
        
        # Отправляем аудио чанками, чтобы избежать ограничений размера сообщения
        chunk_size = 16384  # 16KB чанки
        for i in range(0, len(audio_content), chunk_size):
            chunk = audio_content[i:i+chunk_size]
            
            # Кодируем в base64
            audio_b64 = base64.b64encode(chunk).decode('utf-8')
            
            # Отправляем чанк
            await client_connections[client_id]["openai_ws"].send(json.dumps({
                "type": "input_audio_buffer.append",
                "event_id": f"audio_{time.time()}",
                "audio": audio_b64
            }))
        
        # Коммитим буфер
        await client_connections[client_id]["openai_ws"].send(json.dumps({
            "type": "input_audio_buffer.commit",
            "event_id": f"commit_{time.time()}"
        }))
        
        # Отправляем команду для создания ответа
        await client_connections[client_id]["openai_ws"].send(json.dumps({
            "type": "response.create",
            "event_id": f"resp_{time.time()}",
            "response": {
                "temperature": 0.7
            }
        }))
        
        # Ждем ответа с таймаутом
        try:
            await asyncio.wait_for(
                client_connections[client_id]["response_ready"].wait(),
                timeout=60.0  # Максимальное время ожидания - 60 секунд
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Превышено время ожидания ответа")
        
        # Получаем ответ из очереди
        response_data = await client_connections[client_id]["messages_queue"].get()
        
        if response_data["type"] == "error":
            raise HTTPException(status_code=500, detail=response_data["message"])
        
        # Получаем текст и аудио
        response_text = response_data["text"]
        audio_chunks = response_data["audio"]
        
        # Объединяем аудио чанки (если есть)
        response_audio = None
        if audio_chunks:
            response_audio = "".join(audio_chunks)
        
        # Добавляем сообщения в историю
        user_message = {
            "role": "user",
            "content": "[Аудио сообщение]",
            "timestamp": time.time()
        }
        session.messages.append(user_message)
        
        assistant_message = {
            "role": "assistant",
            "content": response_text,
            "timestamp": time.time()
        }
        session.messages.append(assistant_message)
        
        return {
            "status": "success",
            "response": {
                "text": response_text,
                "audio": response_audio
            }
        }
    
    except Exception as e:
        logger.error(f"Ошибка при обработке аудио: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

# Удаляем сессию
@app.delete("/api/v1/session/{session_id}")
async def delete_session(session_id: str):
    if session_id not in api_sessions:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    
    # Удаляем сессию
    del api_sessions[session_id]
    
    # Также закрываем соединение с OpenAI, если оно существует
    for client_id, client_data in list(client_connections.items()):
        if hasattr(client_data, 'api_session_id') and client_data.api_session_id == session_id:
            # Закрываем соединение
            if client_data["openai_ws"]:
                try:
                    await client_data["openai_ws"].close()
                except:
                    pass
            
            # Удаляем информацию о клиенте
            client_connections[client_id]["active"] = False
            del client_connections[client_id]
    
    return {
        "status": "success",
        "message": "Сессия удалена"
    }

# API прокси для оригинального виджета
@app.get("/api/v1.0/chatgpt_widget_dialog_api")
async def proxy_widget_dialog_api(request: Request):
    """
    Прокси для оригинального виджета. Перехватывает запросы от iframe 
    и обрабатывает их через ваш API вместо оригинального сервера.
    """
    # Получаем параметры запроса
    params = dict(request.query_params)
    
    # Создаем HTML с вашим виджетом
    # Здесь мы заменяем доступ к оригинальному API на ваш собственный
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>WellcomeAI Widget</title>
        <style>
            body, html {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
            }}
            #assistant-container {{
                width: 100%;
                height: 100%;
                display: flex;
                flex-direction: column;
            }}
            #widget-frame {{
                width: 100%;
                height: 100%;
                border: none;
            }}
        </style>
    </head>
    <body>
        <div id="assistant-container">
            <!-- Здесь мы встраиваем ваш собственный интерфейс вместо оригинального -->
            <iframe id="widget-frame" src="/widget" allow="microphone;autoplay"></iframe>
        </div>
        
        <script>
            // Здесь можно добавить JavaScript для обработки сообщений между iframe и родительской страницей
            window.addEventListener('message', function(event) {{
                // Обработка сообщений от родительской страницы или из iframe
                console.log('Received message:', event.data);
                
                // Пример пересылки сообщения в iframe
                const frame = document.getElementById('widget-frame');
                if (frame && event.data && event.data.type) {{
                    frame.contentWindow.postMessage(event.data, '*');
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    # Возвращаем HTML страницу
    return HTMLResponse(content=html_content)

# Путь к файлу widget.js
@app.get("/widget.js", response_class=Response)
async def get_widget_js():
    """Возвращает JavaScript файл с кодом виджета"""
    widget_js_path = os.path.join(os.getcwd(), "widget.js")
    
    # Проверяем наличие файла в корне
    if not os.path.exists(widget_js_path):
        # Проверяем в директории static
        widget_js_path = os.path.join(static_dir, "widget.js")
        if not os.path.exists(widget_js_path):
            # Если файл не найден нигде, возвращаем ошибку или пустой скрипт
            logger.warning("Файл widget.js не найден")
            return Response(
                content="// WellcomeAI Widget not found", 
                media_type="application/javascript"
            )
    
    # Читаем содержимое файла
    try:
        with open(widget_js_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Возвращаем с правильным Content-Type для JavaScript
        return Response(content=content, media_type="application/javascript")
    except Exception as e:
        logger.error(f"Ошибка при чтении файла widget.js: {str(e)}")
        return Response(
            content=f"// Error loading widget: {str(e)}", 
            media_type="application/javascript"
        )

# Дополнительный эндпоинт для проверки статуса виджета
@app.get("/api/widget/status")
async def widget_status(request: Request):
    """Проверка статуса виджета"""
    # Определяем базовый URL для виджета
    base_url = str(request.base_url)
    
    return {
        "status": "active",
        "version": "1.0.0",
        "widget_url": f"{base_url}widget.js",
        "api_healthy": OPENAI_API_KEY is not None,
        "available_voices": AVAILABLE_VOICES
    }

# API-эндпоинт для получения кода для вставки виджета на сайт
@app.get("/api/widget/embed")
async def get_widget_embed_code(request: Request):
    """Возвращает HTML-код для встраивания виджета на сайт"""
    # Определяем базовый URL для виджета
    base_url = str(request.base_url)
    
    # Формируем HTML-код для вставки
    embed_code = f"""
<!-- WellcomeAI Widget Code -->
<script src="{base_url}widget.js" defer></script>
<!-- End WellcomeAI Widget Code -->
"""
    
    return {
        "status": "success",
        "embed_code": embed_code,
        "instructions": "Скопируйте этот код и вставьте его перед закрывающим тегом </body> на вашем сайте."
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Запуск сервера на порту {PORT}")
    # Оптимизированные настройки uvicorn для более быстрой обработки запросов
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        log_level="info",
        timeout_keep_alive=120,  # Увеличенный таймаут для длинных ответов
        loop="auto"              # Использовать оптимальный цикл событий для платформы
    )
