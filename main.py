import os
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis")

# Загружаем переменные окружения
load_dotenv()

# Конфигурация
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PORT = int(os.getenv('PORT', 5050))
REALTIME_WS_URL = 'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01'
SYSTEM_MESSAGE = (
    "Ты Джарвис - умный голосовой помощник. Отвечай коротко и по существу. "
    "Ты готов помочь пользователю с любыми вопросами и задачами."
)
VOICE = 'alloy'  # Доступные голоса: alloy, echo, fable, onyx, nova, shimmer

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

# Отслеживаемые события от OpenAI для логирования
LOG_EVENT_TYPES = [
    'response.content.done', 'rate_limits.updated', 'response.done',
    'input_audio_buffer.committed', 'input_audio_buffer.speech_stopped',
    'input_audio_buffer.speech_started', 'session.created'
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
        # Проверим, что ключ API существует
        if not OPENAI_API_KEY:
            return {
                "status": "error",
                "message": "API ключ не настроен на сервере"
            }
            
        # Возвращаем успешный статус
        return {
            "status": "success",
            "api_key_valid": True,
            "openai_api_version": "Realtime API"
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
        "active": True
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
        await send_session_update(openai_ws)
        
        # Создаем две задачи для двустороннего обмена сообщениями
        client_to_openai = asyncio.create_task(forward_client_to_openai(websocket, openai_ws, client_id))
        openai_to_client = asyncio.create_task(forward_openai_to_client(openai_ws, websocket, client_id))
        
        # Ждем, пока одна из задач не завершится
        _, pending = await asyncio.wait(
            [client_to_openai, openai_to_client],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Отменяем оставшиеся задачи
        for task in pending:
            task.cancel()
        
    except Exception as e:
        logger.error(f"Ошибка при обработке WebSocket соединения: {str(e)}")
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
            # Получаем сообщение от клиента
            message = await client_ws.receive_text()
            
            # Парсим JSON
            try:
                data = json.loads(message)
                msg_type = data.get("type", "unknown")
                
                # Выводим информацию о сообщении в логи
                logger.debug(f"[Клиент -> OpenAI] {msg_type}")
                
                # Отправляем сообщение в OpenAI
                await openai_ws.send(message)
                
            except json.JSONDecodeError:
                logger.error(f"Получены некорректные данные от клиента: {message[:100]}...")
    
    except WebSocketDisconnect:
        logger.info(f"Клиент {client_id} отключился")
    except Exception as e:
        logger.error(f"Ошибка при пересылке данных от клиента к OpenAI: {str(e)}")

async def forward_openai_to_client(openai_ws, client_ws: WebSocket, client_id: int):
    """Пересылает сообщения от API OpenAI клиенту (браузеру)"""
    try:
        async for openai_message in openai_ws:
            if client_id not in client_connections or not client_connections[client_id]["active"]:
                break
                
            try:
                # Парсим JSON от OpenAI
                response = json.loads(openai_message)
                
                # Логируем определенные типы событий
                if response.get('type') in LOG_EVENT_TYPES:
                    logger.info(f"[OpenAI -> Клиент] {response.get('type')}")
                
                # Пересылаем сообщение клиенту
                await client_ws.send_text(openai_message)
                
            except json.JSONDecodeError:
                logger.error(f"Получены некорректные данные от OpenAI: {openai_message[:100]}...")
    
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
        logger.error(f"Ошибка при пересылке данных от OpenAI к клиенту: {str(e)}")

async def send_session_update(openai_ws):
    """Отправляет настройки сессии в WebSocket OpenAI"""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "pcm16",  # Формат входящего аудио
            "output_audio_format": "pcm16", # Формат исходящего аудио
            "voice": VOICE,                 # Голос ассистента
            "instructions": SYSTEM_MESSAGE, # Системное сообщение
            "modalities": ["text", "audio"],# Поддерживаемые модальности
            "temperature": 0.8,             # Температура генерации
        }
    }
    logger.info(f"Отправка настроек сессии: {json.dumps(session_update)}")
    await openai_ws.send(json.dumps(session_update))

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Запуск сервера на порту {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
