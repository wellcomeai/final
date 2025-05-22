import os
import sys
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import uvicorn

from websocket_handler import handle_websocket_connection

# Загрузка переменных окружения
load_dotenv()

# Создание FastAPI приложения
app = FastAPI(title="Voice Assistant", version="1.0.0")

# Настройка CORS для Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене можно ограничить
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение статических файлов
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Главная точка подключения WebSocket для голосового ассистента
    """
    await handle_websocket_connection(websocket)

@app.get("/")
async def root():
    """Главная страница - отдаем HTML клиент"""
    return FileResponse('static/index.html')

@app.get("/health")
async def health_check():
    """Health check для Render"""
    return {"status": "healthy", "service": "voice-assistant"}

if __name__ == "__main__":
    # Получаем порт из переменных окружения (Render автоматически устанавливает PORT)
    port = int(os.getenv("PORT", 8000))
    
    # В продакшене используем uvicorn напрямую
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        # Отключаем reload в продакшене
        reload=False
    )
