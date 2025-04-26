import os
import asyncio
import json
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import websockets

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

# Получаем API ключ из переменных окружения
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY не задан в переменных окружения")

# URL для создания сессии Realtime API
REALTIME_SESSION_URL = "https://api.openai.com/v1/realtime/sessions"
REALTIME_WEBSOCKET_URL = "wss://api.openai.com/v1/realtime/conversations"

# Глобальный кеш для хранения созданных сессий
sessions = {}

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/create-session")
async def create_session():
    """Создает новую сессию в OpenAI Realtime API и возвращает токен клиенту"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                REALTIME_SESSION_URL,
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": "gpt-4o-realtime-preview",
                    "modalities": ["audio", "text"],
                    "voice": "alloy",  # Можно выбрать из: alloy, echo, fable, onyx, nova, shimmer
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
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Ошибка создания сессии в OpenAI")
            
            session_data = response.json()
            session_id = session_data["id"]
            client_secret = session_data["client_secret"]["value"]
            
            # Сохраняем сессию в кеше
            sessions[session_id] = {
                "client_secret": client_secret,
                "session_data": session_data
            }
            
            return {
                "session_id": session_id,
                "client_secret": client_secret
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при создании сессии: {str(e)}")

@app.websocket("/ws-proxy/{session_id}")
async def websocket_proxy(websocket: WebSocket, session_id: str):
    """
    Прокси для WebSocket соединения между клиентом и OpenAI Realtime API
    """
    if session_id not in sessions:
        await websocket.close(code=1008, reason="Сессия не найдена")
        return
    
    session_info = sessions[session_id]
    client_secret = session_info["client_secret"]
    
    await websocket.accept()
    print(f"WebSocket соединение принято для сессии {session_id}")
    
    # Устанавливаем соединение с OpenAI через websockets
    try:
        async with websockets.connect(
            f"{REALTIME_WEBSOCKET_URL}/{session_id}",
            extra_headers={"Authorization": f"Bearer {client_secret}"}
        ) as ws_openai:
            # Создаем две задачи для двустороннего обмена данными
            async def forward_to_openai():
                try:
                    while True:
                        message = await websocket.receive_text()
                        print(f"-> OpenAI: {message[:100]}...")
                        await ws_openai.send(message)
                except WebSocketDisconnect:
                    print(f"Клиент отключился от сессии {session_id}")
                except Exception as e:
                    print(f"Ошибка при отправке в OpenAI: {str(e)}")
            
            async def forward_from_openai():
                try:
                    while True:
                        message = await ws_openai.recv()
                        print(f"<- OpenAI: {message[:100] if isinstance(message, str) else 'binary data'}...")
                        if isinstance(message, str):
                            await websocket.send_text(message)
                        else:
                            await websocket.send_bytes(message)
                except websockets.exceptions.ConnectionClosed:
                    print(f"OpenAI закрыл соединение для сессии {session_id}")
                except Exception as e:
                    print(f"Ошибка при получении от OpenAI: {str(e)}")
            
            # Запускаем обе задачи одновременно
            forwarding_task1 = asyncio.create_task(forward_to_openai())
            forwarding_task2 = asyncio.create_task(forward_from_openai())
            
            # Ждем, пока одна из задач не завершится (это произойдет при разрыве соединения)
            done, pending = await asyncio.wait(
                [forwarding_task1, forwarding_task2],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Отменяем незавершенные задачи
            for task in pending:
                task.cancel()
                
    except Exception as e:
        print(f"Ошибка при соединении с OpenAI: {str(e)}")
        await websocket.close(code=1011, reason=f"Ошибка: {str(e)}")

# Для обратной совместимости с предыдущей версией
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        await websocket.send_text("Эта версия API обновлена. Пожалуйста, используйте новый интерфейс для голосового общения.")
        await websocket.close()
    except WebSocketDisconnect:
        print("Клиент отключился")

# Запускаем приложение с uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
