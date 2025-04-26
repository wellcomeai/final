import os
import json
import base64
import asyncio
import logging
import traceback
import websockets
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("claude-voice")
logger.setLevel(logging.DEBUG)

# Загружаем переменные окружения
load_dotenv()

# Конфигурация
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PORT = int(os.getenv('PORT', 5050))
REALTIME_WS_URL = 'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01'
SYSTEM_MESSAGE = (
    "Ты Claude Sonnet - умный голосовой помощник. Отвечай на вопросы пользователя коротко, "
    "информативно и с небольшой ноткой юмора, когда это уместно. Стремись быть полезным "
    "и предоставлять точную информацию."
)

# Допустимые голоса
AVAILABLE_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
DEFAULT_VOICE = "alloy"

app = FastAPI()

# Добавляем CORS middleware для разрешения кросс-доменных запросов
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Проверяем наличие директории static и создаем ее, если она не существует
static_dir = os.path.join(os.getcwd(), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
    with open(os.path.join(static_dir, "index.html"), "w") as f:
        f.write("<html><body><h1>Placeholder</h1></body></html>")
    logger.info(f"Создана директория static с заглушкой index.html")

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

if not OPENAI_API_KEY:
    raise ValueError('Отсутствует ключ API OpenAI. Пожалуйста, укажите его в .env файле.')

# Отслеживаемые события от OpenAI для подробного логирования
LOG_EVENT_TYPES = [
    'response.content.done', 'rate_limits.updated', 'response.done',
    'input_audio_buffer.committed', 'input_audio_buffer.speech_stopped',
    'input_audio_buffer.speech_started', 'session.created', 'session.updated'
]

# Хранилище активных соединений клиент <-> OpenAI
client_connections = {}

@app.get("/")
async def index_page():
    """Возвращает HTML страницу с интерфейсом голосового помощника"""
    try:
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r") as file:
                content = file.read()
            return HTMLResponse(content=content)
        else:
            logger.warning(f"Файл {index_path} не найден")
            return {"message": "Файл index.html не найден в директории static"}
    except Exception as e:
        logger.error(f"Ошибка при отдаче главной страницы: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Ошибка сервера: {str(e)}"}
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
            "available_voices": AVAILABLE_VOICES
        }
    except Exception as e:
        logger.error(f"Ошибка при проверке API: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

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
        "turn_detection_enabled": True
    }
    
    try:
        # Устанавливаем соединение с OpenAI
        openai_ws = await websockets.connect(
            REALTIME_WS_URL,
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            }
        )
        
        client_connections[client_id]["openai_ws"] = openai_ws
        logger.info(f"Соединение с OpenAI установлено для клиента {client_id}")
        
        # Отправляем настройки сессии в OpenAI
        await send_session_update(openai_ws, DEFAULT_VOICE)
        
        # Создаем две задачи для двустороннего обмена сообщениями
        client_to_openai = asyncio.create_task(forward_client_to_openai(websocket, openai_ws, client_id))
        openai_to_client = asyncio.create_task(forward_openai_to_client(openai_ws, websocket, client_id))
        
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
                logger.error(f"Задача {task.get_name()} завершилась с ошибкой: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Отменяем оставшиеся задачи
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Ошибка при отмене задачи: {str(e)}")
        
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
        
        # Удаляем информацию о клиенте
        if client_id in client_connections:
            client_connections[client_id]["active"] = False
            del client_connections[client_id]
        
        logger.info(f"Соединение с клиентом {client_id} закрыто")

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
                
            # Парсим JSON
            try:
                data = json.loads(message)
                msg_type = data.get("type", "unknown")
                
                # Если это обновление сессии, обрабатываем его
                if msg_type == "session.update":
                    # Обновляем настройки сессии
                    session_data = data.get("session", {})
                    
                    # Если указан голос, обновляем его
                    if "voice" in session_data:
                        voice = session_data["voice"]
                        if voice in AVAILABLE_VOICES:
                            client_connections[client_id]["voice"] = voice
                            logger.info(f"Голос для клиента {client_id} изменен на {voice}")
                
                # Выводим информацию о сообщении в логи
                logger.debug(f"[Клиент {client_id} -> OpenAI] {msg_type}")
                
                # Отправляем сообщение в OpenAI
                await openai_ws.send(message)
                
            except json.JSONDecodeError as e:
                logger.error(f"Получены некорректные данные от клиента: {message[:100]}...")
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
        logger.info(f"Соединение с OpenAI закрыто для клиента {client_id}: {e.code}, {e.reason}")
        try:
            # Сообщаем клиенту о закрытии соединения
            await client_ws.send_json({
                "type": "error",
                "error": {
                    "message": f"Соединение с OpenAI закрыто: {e.reason}"
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
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500
    } if turn_detection_enabled else None
    
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": turn_detection,
            "input_audio_format": "pcm16",      # Формат входящего аудио
            "output_audio_format": "pcm16",     # Формат исходящего аудио
            "voice": voice,                     # Голос ассистента
            "instructions": SYSTEM_MESSAGE,     # Системное сообщение
            "modalities": ["text", "audio"],    # Поддерживаемые модальности
            "temperature": 0.7,                 # Температура генерации
        }
    }
    
    logger.info(f"Отправка настроек сессии с голосом {voice}")
    await openai_ws.send(json.dumps(session_update))

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Запуск сервера на порту {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
