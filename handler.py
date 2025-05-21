from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio
import uuid
import base64
from websockets.exceptions import ConnectionClosed
from typing import Dict, List

from client import SimpleOpenAIClient
from audio_utils import base64_to_audio_buffer
from config import settings

# Словарь активных соединений по assistant_id
active_connections: Dict[str, List[WebSocket]] = {}

async def handle_websocket_connection(
    websocket: WebSocket,
    assistant_id: str
) -> None:
    """Обрабатывает WebSocket соединение с клиентом"""
    client_id = str(uuid.uuid4())
    openai_client = None

    try:
        await websocket.accept()
        print(f"WebSocket соединение принято: client_id={client_id}, assistant_id={assistant_id}")

        # Регистрируем соединение
        active_connections.setdefault(assistant_id, []).append(websocket)

        # Создаем простую конфигурацию ассистента (в реальном проекте можно загружать из БД)
        assistant_config = {
            "name": "Mini Voice Assistant",
            "voice": "alloy",
            "language": "ru",
            "system_prompt": "Ты полезный голосовой ассистент. Отвечай кратко и по делу на русском языке.",
            "functions": []  # Здесь можно добавить функции
        }

        # Используем API ключ из переменной окружения
        api_key = settings.OPENAI_API_KEY
        
        if not api_key:
            await websocket.send_json({
                "type": "error",
                "error": {"code": "no_api_key", "message": "Отсутствует ключ API OpenAI"}
            })
            await websocket.close(code=1008)
            return

        # Подключаемся к OpenAI
        openai_client = SimpleOpenAIClient(api_key=api_key, assistant_config=assistant_config)
        if not await openai_client.connect():
            await websocket.send_json({
                "type": "error",
                "error": {"code": "openai_connection_failed", "message": "Не удалось подключиться к OpenAI"}
            })
            await websocket.close(code=1008)
            return

        # Сообщаем клиенту об успешном подключении
        await websocket.send_json({"type": "connection_status", "status": "connected", "message": "Соединение установлено"})

        audio_buffer = bytearray()
        is_processing = False

        # Запускаем приём сообщений от OpenAI
        openai_task = asyncio.create_task(handle_openai_messages(openai_client, websocket))

        # Основной цикл приёма от клиента
        while True:
            try:
                message = await websocket.receive()

                if "text" in message:
                    data = json.loads(message["text"])
                    msg_type = data.get("type", "")

                    if msg_type == "ping":
                        await websocket.send_json({"type": "pong"})
                        continue

                    if msg_type == "input_audio_buffer.append":
                        audio_chunk = base64_to_audio_buffer(data["audio"])
                        audio_buffer.extend(audio_chunk)
                        if openai_client.is_connected:
                            await openai_client.process_audio(audio_chunk)
                        await websocket.send_json({"type": "input_audio_buffer.append.ack", "event_id": data.get("event_id")})
                        continue

                    if msg_type == "input_audio_buffer.commit" and not is_processing:
                        is_processing = True
                        
                        # Проверка минимального размера буфера
                        if not audio_buffer or len(audio_buffer) < 3200:  
                            await websocket.send_json({
                                "type": "warning",
                                "warning": {"code": "audio_buffer_too_small", "message": "Аудио слишком короткое, попробуйте говорить дольше"}
                            })
                            audio_buffer.clear()
                            is_processing = False
                            continue

                        if openai_client.is_connected:
                            await openai_client.commit_audio()
                            await websocket.send_json({"type": "input_audio_buffer.commit.ack", "event_id": data.get("event_id")})
                        else:
                            await websocket.send_json({
                                "type": "error",
                                "error": {"code": "openai_not_connected", "message": "Соединение с OpenAI потеряно"}
                            })

                        audio_buffer.clear()
                        is_processing = False
                        continue

                    if msg_type == "input_audio_buffer.clear":
                        audio_buffer.clear()
                        if openai_client.is_connected:
                            await openai_client.clear_audio_buffer()
                        await websocket.send_json({"type": "input_audio_buffer.clear.ack", "event_id": data.get("event_id")})
                        continue

                    if msg_type == "response.cancel":
                        if openai_client.is_connected:
                            await openai_client.cancel_response()
                        await websocket.send_json({"type": "response.cancel.ack", "event_id": data.get("event_id")})
                        continue

                    # Любые остальные типы
                    await websocket.send_json({
                        "type": "error",
                        "error": {"code": "unknown_message_type", "message": f"Неизвестный тип сообщения: {msg_type}"}
                    })

                elif "bytes" in message:
                    # raw-байты от клиента
                    audio_buffer.extend(message["bytes"])
                    await websocket.send_json({"type": "binary.ack"})

            except (WebSocketDisconnect, ConnectionClosed):
                break
            except Exception as e:
                print(f"Ошибка в цикле WebSocket: {e}")
                break

        # завершение
        if not openai_task.done():
            openai_task.cancel()
            await asyncio.sleep(0)  # даём задаче отмениться

    finally:
        if openai_client:
            await openai_client.close()
        # убираем из active_connections
        conns = active_connections.get(assistant_id, [])
        if websocket in conns:
            conns.remove(websocket)
        print(f"Удалено WebSocket соединение: client_id={client_id}")


async def handle_openai_messages(openai_client: SimpleOpenAIClient, websocket: WebSocket):
    """Обрабатывает сообщения от OpenAI и передает их клиенту"""
    if not openai_client.is_connected or not openai_client.ws:
        print("OpenAI клиент не подключен.")
        return
    
    try:
        print(f"Начало обработки сообщений от OpenAI для клиента {openai_client.client_id}")
        
        while True:
            try:
                raw = await openai_client.ws.recv()
                
                try:
                    response_data = json.loads(raw)
                except json.JSONDecodeError:
                    print(f"Ошибка декодирования JSON: {raw[:200]}")
                    continue
                    
                # Логирование типа полученного сообщения
                msg_type = response_data.get("type", "unknown")
                print(f"Получено сообщение от OpenAI: тип={msg_type}")
                
                # Сохраняем ID ответа, если это начало ответа
                if msg_type == "response.started" and "response_id" in response_data:
                    openai_client.current_response_id = response_data["response_id"]
                    print(f"Сохранен response_id: {openai_client.current_response_id}")
                
                # Сбрасываем ID ответа, если ответ завершен
                if msg_type == "response.done":
                    openai_client.current_response_id = None
                
                # Обработка ошибок
                if msg_type == "error":
                    print(f"ОШИБКА API: {json.dumps(response_data, ensure_ascii=False)}")
                    await websocket.send_json(response_data)
                    continue
                
                # Если это аудио-чанк — отдаём как bytes
                if msg_type == "audio":
                    b64 = response_data.get("data", "")
                    chunk = base64.b64decode(b64)
                    await websocket.send_bytes(chunk)
                    continue
                
                # Все остальные сообщения отправляем как JSON
                await websocket.send_json(response_data)

            except ConnectionClosed as e:
                print(f"Соединение с OpenAI закрыто: {e}")
                # В минимальной версии просто прерываем обработку
                await websocket.send_json({
                    "type": "error",
                    "error": {"code": "openai_connection_lost", "message": "Соединение с AI потеряно"}
                })
                break

    except (ConnectionClosed, asyncio.CancelledError):
        print(f"Соединение закрыто для клиента {openai_client.client_id}")
        return
    except Exception as e:
        print(f"Ошибка в обработчике сообщений OpenAI: {e}")
