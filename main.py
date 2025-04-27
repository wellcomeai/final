import os
import json
import base64
import asyncio
import logging
import traceback
import shutil
import websockets
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

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
        await asyncio.sleep(0.5)
        
        # Создаем новое соединение
        new_openai_ws = await create_openai_connection()
        client_connections[client_id]["openai_ws"] = new_openai_ws
        
        # Отправляем настройки сессии с новым голосом
        await send_session_update(new_openai_ws, new_voice)
        
        # Создаем новые задачи
        client_to_openai = asyncio.create_task(forward_client_to_openai(client_ws, new_openai_ws, client_id))
        openai_to_client = asyncio.create_task(forward_openai_to_client(new_openai_ws, client_ws, client_id))
        
        # Сохраняем новые задачи
        client_connections[client_id]["tasks"] = [client_to_openai, openai_to_client]
        client_connections[client_id]["voice"] = new_voice
        
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
                            # Клиент не должен видеть эту ошибку
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
            "max_response_output_tokens": 500   # Лимит токенов для ответа
        }
    }
    
    logger.info(f"Отправка настроек сессии с голосом {voice}")
    try:
        await openai_ws.send(json.dumps(session_update))
    except Exception as e:
        logger.error(f"Ошибка при отправке настроек сессии: {str(e)}")
        raise

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
