import os
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from base64 import b64encode, b64decode
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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
    Это нужно, если у вас проблемы с CORS или вы хотите логировать/модифицировать сообщения
    """
    if session_id not in sessions:
        await websocket.close(code=1008, reason="Сессия не найдена")
        return
    
    session_info = sessions[session_id]
    client_secret = session_info["client_secret"]
    
    await websocket.accept()
    
    # Создаем клиентское соединение к OpenAI Realtime API
    async with httpx.AsyncClient() as client:
        async with client.websocket(
            f"{REALTIME_WEBSOCKET_URL}/{session_id}",
            headers={"Authorization": f"Bearer {client_secret}"}
        ) as ws_openai:
            # Задачи для одновременного обмена сообщениями в обоих направлениях
            async def forward_to_openai():
                try:
                    while True:
                        data = await websocket.receive_bytes()
                        await ws_openai.send_bytes(data)
                except Exception as e:
                    print(f"Ошибка при отправке в OpenAI: {str(e)}")
            
            async def forward_to_client():
                try:
                    while True:
                        data = await ws_openai.receive_bytes()
                        await websocket.send_bytes(data)
                except Exception as e:
                    print(f"Ошибка при отправке клиенту: {str(e)}")
            
            # Запускаем обе задачи одновременно
            forward_tasks = [
                asyncio.create_task(forward_to_openai()),
                asyncio.create_task(forward_to_client())
            ]
            
            # Ждем, пока одна из задач не завершится (это произойдет при разрыве соединения)
            try:
                await asyncio.wait(
                    forward_tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )
            except Exception as e:
                print(f"Ошибка WebSocket proxy: {str(e)}")
            finally:
                # Отменяем все задачи при завершении
                for task in forward_tasks:
                    if not task.done():
                        task.cancel()

# Запускаем приложение с uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
