import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai import AsyncOpenAI

# Инициализация клиента OpenAI
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

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
        await websocket.send_text(f"Ошибка: {str(e)}")
