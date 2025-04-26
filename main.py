import os
import asyncio
import json
import logging
import traceback
import urllib.parse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis")
logger.setLevel(logging.DEBUG)

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

# Получаем API ключ из переменных окружения
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY не задан в переменных окружения")
    raise ValueError("OPENAI_API_KEY не задан в переменных окружения")
else:
    # Маскируем ключ для логов, показывая только первые и последние 4 символа
    masked_key = OPENAI_API_KEY[:4] + "..." + OPENAI_API_KEY[-4:]
    logger.info(f"OPENAI_API_KEY получен: {masked_key}")

# URL для создания сессии Realtime API
REALTIME_SESSION_URL = "https://api.openai.com/v1/realtime/sessions"
REALTIME_WEBSOCKET_URL = "wss://api.openai.com/v1/realtime/conversations"

# Глобальный кеш для хранения созданных сессий
sessions = {}

# Ошибка для клиента при неправильных заголовках
@app.exception_handler(Exception)
async def handle_exception(request: Request, exc: Exception):
    logger.error(f"Необработанное исключение: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"message": f"Внутренняя ошибка сервера: {str(exc)}"}
    )

@app.get("/")
async def root():
    """Корневой маршрут, возвращает HTML-страницу"""
    try:
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            logger.info("Запрошена главная страница")
            return FileResponse(index_path)
        else:
            logger.warning(f"Файл {index_path} не найден")
            return {"message": "Файл index.html не найден в директории static"}
    except Exception as e:
        logger.error(f"Ошибка при отдаче главной страницы: {str(e)}")
        raise

@app.get("/api/check")
async def check_api():
    """Маршрут для проверки доступности API и валидности ключа"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}
            )
            
            if response.status_code == 200:
                models = response.json()
                # Проверяем наличие модели gpt-4o-realtime-preview в списке
                has_realtime = any("gpt-4o-realtime-preview" in model.get("id", "") for model in models.get("data", []))
                
                return {
                    "status": "success",
                    "api_key_valid": True,
                    "has_realtime_access": has_realtime,
                    "models_count": len(models.get("data", []))
                }
            else:
                logger.warning(f"Ошибка проверки API: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "api_key_valid": False,
                    "error": response.text
                }
    except Exception as e:
        logger.error(f"Ошибка при проверке API: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/create-session")
async def create_session():
    """Создает новую сессию в OpenAI Realtime API и возвращает токен клиенту"""
    try:
        logger.info("Запрос на создание новой сессии")
        
        # Проверяем наличие ключа API
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY не задан")
            raise HTTPException(status_code=500, detail="API ключ не настроен на сервере")
        
        # Параметры сессии
        session_params = {
            "model": "gpt-4o-realtime-preview",
            "modalities": ["audio", "text"],
            "voice": "alloy",
            "instructions": "Ты Джарвис - умный голосовой помощник. Отвечай коротко и по существу.",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": "whisper-1"
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500
            }
        }
        
        logger.debug(f"Параметры сессии: {json.dumps(session_params)}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.debug("Отправка запроса на создание сессии в OpenAI...")
            response = await client.post(
                REALTIME_SESSION_URL,
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json=session_params
            )
            
            logger.debug(f"Получен ответ от OpenAI: статус {response.status_code}")
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"Ошибка создания сессии в OpenAI: {response.status_code} - {error_text}")
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"Ошибка создания сессии в OpenAI: {error_text}"
                )
            
            session_data = response.json()
            logger.debug(f"Данные сессии: {json.dumps(session_data)}")
            
            session_id = session_data["id"]
            client_secret = session_data["client_secret"]["value"]
            
            # Сохраняем сессию в кеше
            sessions[session_id] = {
                "client_secret": client_secret,
                "session_data": session_data,
                "created_at": asyncio.get_event_loop().time()
            }
            
            logger.info(f"Сессия создана успешно, ID: {session_id}")
            
            return {
                "session_id": session_id,
                "client_secret": client_secret
            }
    except HTTPException as he:
        # Пробрасываем HTTP исключения дальше
        logger.error(f"HTTP ошибка при создании сессии: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Неожиданная ошибка при создании сессии: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Ошибка при создании сессии: {str(e)}")

@app.get("/direct-connect/{session_id}")
async def direct_connect(session_id: str):
    """Возвращает URL и токен для прямого подключения к API OpenAI (обходной путь)"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    
    session_info = sessions[session_id]
    client_secret = session_info["client_secret"]
    
    return {
        "websocket_url": f"{REALTIME_WEBSOCKET_URL}/{session_id}",
        "authorization": f"Bearer {client_secret}"
    }

@app.websocket("/ws-proxy/{session_id}")
async def websocket_proxy(websocket: WebSocket, session_id: str):
    """
    Прокси для WebSocket соединения между клиентом и OpenAI Realtime API
    Этот подход избегает прямого использования библиотеки websockets для соединения с OpenAI
    """
    client_ip = websocket.client.host
    logger.info(f"Новое WebSocket соединение от {client_ip} для сессии {session_id}")
    
    # Проверка наличия сессии
    if session_id not in sessions:
        logger.warning(f"Сессия {session_id} не найдена в кеше")
        await websocket.accept()
        await websocket.send_text(json.dumps({
            "type": "error",
            "error": {
                "message": "Сессия не найдена или истекла. Пожалуйста, создайте новую сессию."
            }
        }))
        await websocket.close(code=1008, reason="Сессия не найдена")
        return
    
    # Получаем данные сессии
    session_info = sessions[session_id]
    client_secret = session_info["client_secret"]
    
    # Принимаем соединение
    await websocket.accept()
    logger.info(f"WebSocket соединение принято для сессии {session_id}")
    
    # Отправляем инструкции клиенту для прямого подключения к OpenAI
    await websocket.send_text(json.dumps({
        "type": "connection_info",
        "websocket_url": f"{REALTIME_WEBSOCKET_URL}/{session_id}",
        "authorization": f"Bearer {client_secret}"
    }))
    
    # Сообщаем клиенту, что нужно использовать прямое подключение
    await websocket.send_text(json.dumps({
        "type": "error",
        "error": {
            "code": "direct_connection_required",
            "message": "Пожалуйста, используйте прямое подключение к OpenAI API. Смотрите инструкции в консоли."
        }
    }))
    
    # Поддерживаем соединение открытым для отладки
    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"Получено сообщение от клиента: {data[:100]}...")
            # Эхо-ответ для подтверждения работы соединения
            await websocket.send_text(json.dumps({
                "type": "echo",
                "message": "Сервер получил сообщение, но работает в режиме прямого подключения"
            }))
    except WebSocketDisconnect:
        logger.info(f"Клиент отключился от сессии {session_id}")
    except Exception as e:
        logger.error(f"Ошибка в WebSocket: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        await websocket.close()
        logger.info(f"WebSocket соединение закрыто для сессии {session_id}")

# Для обратной совместимости с предыдущей версией
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Обработчик для совместимости со старым интерфейсом"""
    client_ip = websocket.client.host
    logger.info(f"Подключение к старому WebSocket эндпоинту от {client_ip}")
    
    try:
        await websocket.accept()
        logger.info(f"Соединение со старым WebSocket принято")
        
        # Отправляем сообщение о необходимости обновления
        await websocket.send_text("Эта версия API обновлена. Пожалуйста, используйте новый голосовой интерфейс на главной странице.")
        
        # Ждем немного перед закрытием
        await asyncio.sleep(1)
        await websocket.close()
        
    except WebSocketDisconnect:
        logger.info("Клиент отключился от старого WebSocket")
    except Exception as e:
        logger.error(f"Ошибка в старом WebSocket: {str(e)}")
        logger.error(traceback.format_exc())

# Запускаем приложение с uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Запуск сервера на порту {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
