import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai import AsyncOpenAI
from fastapi.middleware.cors import CORSMiddleware

# Инициализация клиента OpenAI
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Добавляем CORS middleware для разрешения кросс-доменных запросов
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем все источники (в продакшене лучше указать конкретные домены)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "GPT-4o WebSocket API работает. Подключайтесь через WebSocket к /ws"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            asyncio.create_task(handle_message(data, websocket))
    except WebSocketDisconnect:
        print("Клиент отключился")

async def handle_message(message: str, websocket: WebSocket):
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": message}],
            stream=True
        )
        async for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    await websocket.send_text(delta)
        await websocket.send_text("[DONE]")
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        await websocket.send_text(f"Ошибка: {str(e)}")
