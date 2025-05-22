import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

from websocket_handler import handle_websocket_connection

# Загрузка переменных окружения
load_dotenv()

# Создание FastAPI приложения
app = FastAPI(title="Voice Assistant", version="1.0.0")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Главная точка подключения WebSocket для голосового ассистента
    """
    await handle_websocket_connection(websocket)

@app.get("/")
async def root():
    return {"message": "Voice Assistant API is running"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=True
    )
