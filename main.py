from fastapi import FastAPI, WebSocket, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from pathlib import Path

from handler import handle_websocket_connection

# Создаем папку static, если её нет
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

app = FastAPI(title="Mini Voice Assistant")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.websocket("/ws/{assistant_id}")
async def websocket_endpoint(websocket: WebSocket, assistant_id: str):
    await handle_websocket_connection(websocket, assistant_id)

@app.get("/")
async def root():
    # Перенаправляем с корня на веб-интерфейс
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")

# Проверка здоровья для Render
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
