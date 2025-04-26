import os
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

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
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o",
            messages=[{"role": "user", "content": message}],
            stream=True
        )
        async for chunk in response:
            if "choices" in chunk:
                delta = chunk["choices"][0]["delta"].get("content")
                if delta:
                    await websocket.send_text(delta)
        await websocket.send_text("[DONE]")
    except Exception as e:
        await websocket.send_text(f"Ошибка: {str(e)}")
