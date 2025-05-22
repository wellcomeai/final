import asyncio
import json
import uuid
import base64
import logging
from fastapi import WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

from openai_client import OpenAIRealtimeClient
from audio_utils import base64_to_audio_buffer

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def handle_websocket_connection(websocket: WebSocket):
    """
    Главный обработчик WebSocket соединения.
    Управляет потоками данных между клиентом и OpenAI.
    """
    client_id = str(uuid.uuid4())
    openai_client = None
    
    try:
        # Принимаем WebSocket соединение
        await websocket.accept()
        logger.info(f"WebSocket соединение принято: client_id={client_id}")
        
        # Получаем API ключ из переменных окружения
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            await websocket.send_json({
                "type": "error",
                "error": {"code": "no_api_key", "message": "OpenAI API key not found"}
            })
            await websocket.close(code=1008)
            return
        
        # Создаем и подключаем клиент OpenAI
        openai_client = OpenAIRealtimeClient(api_key, client_id)
        
        if not await openai_client.connect():
            await websocket.send_json({
                "type": "error", 
                "error": {"code": "openai_connection_failed", "message": "Failed to connect to OpenAI"}
            })
            await websocket.close(code=1008)
            return
        
        # Уведомляем клиента об успешном подключении
        await websocket.send_json({
            "type": "connection_status",
            "status": "connected", 
            "message": "Voice assistant ready"
        })
        
        # Буфер для накопления аудио данных
        audio_buffer = bytearray()
        
        # Запускаем задачу для обработки сообщений от OpenAI
        openai_task = asyncio.create_task(
            handle_openai_messages(openai_client, websocket)
        )
        
        # Основной цикл обработки сообщений от клиента
        while True:
            try:
                # Получаем сообщение от клиента
                message = await websocket.receive()
                
                if "text" in message:
                    # Обрабатываем текстовые сообщения (JSON)
                    data = json.loads(message["text"])
                    msg_type = data.get("type", "")
                    
                    if msg_type == "ping":
                        # Ответ на ping для поддержания соединения
                        await websocket.send_json({"type": "pong"})
                        
                    elif msg_type == "input_audio_buffer.append":
                        # Добавление аудио данных в буфер
                        audio_chunk = base64_to_audio_buffer(data["audio"])
                        audio_buffer.extend(audio_chunk)
                        
                        # Передаем аудио в OpenAI
                        if openai_client.is_connected:
                            await openai_client.send_audio(audio_chunk)
                        
                        # Подтверждаем получение
                        await websocket.send_json({
                            "type": "input_audio_buffer.append.ack",
                            "event_id": data.get("event_id")
                        })
                        
                    elif msg_type == "input_audio_buffer.commit":
                        # Завершение записи аудио (пользователь закончил говорить)
                        
                        # Проверяем минимальный размер аудио
                        if len(audio_buffer) < 3200:  # ~100мс при 16kHz/16bit/mono
                            await websocket.send_json({
                                "type": "warning",
                                "warning": {
                                    "code": "audio_too_short",
                                    "message": "Audio too short, please speak longer"
                                }
                            })
                            audio_buffer.clear()
                            continue
                        
                        # Коммитим аудио в OpenAI
                        if openai_client.is_connected:
                            await openai_client.commit_audio()
                            await websocket.send_json({
                                "type": "input_audio_buffer.commit.ack",
                                "event_id": data.get("event_id")
                            })
                        
                        # Очищаем буфер
                        audio_buffer.clear()
                        
                    elif msg_type == "input_audio_buffer.clear":
                        # Очистка аудио буфера
                        audio_buffer.clear()
                        if openai_client.is_connected:
                            await openai_client.clear_audio_buffer()
                        await websocket.send_json({
                            "type": "input_audio_buffer.clear.ack",
                            "event_id": data.get("event_id")
                        })
                        
                    elif msg_type == "response.cancel":
                        # Отмена текущего ответа
                        if openai_client.is_connected:
                            await openai_client.cancel_response()
                        await websocket.send_json({
                            "type": "response.cancel.ack",
                            "event_id": data.get("event_id")
                        })
                        
                elif "bytes" in message:
                    # Обработка бинарных данных (raw аудио)
                    audio_buffer.extend(message["bytes"])
                    if openai_client.is_connected:
                        await openai_client.send_audio(message["bytes"])
                    
            except (WebSocketDisconnect, ConnectionClosed):
                logger.info(f"WebSocket соединение закрыто: client_id={client_id}")
                break
            except Exception as e:
                logger.error(f"Ошибка в WebSocket цикле: {e}")
                break
        
        # Отменяем задачу обработки OpenAI сообщений
        if not openai_task.done():
            openai_task.cancel()
            try:
                await openai_task
            except asyncio.CancelledError:
                pass
                
    except Exception as e:
        logger.error(f"Критическая ошибка в handle_websocket_connection: {e}")
        
    finally:
        # Закрываем соединение с OpenAI
        if openai_client:
            await openai_client.close()
        logger.info(f"Соединение завершено: client_id={client_id}")


async def handle_openai_messages(openai_client: OpenAIRealtimeClient, websocket: WebSocket):
    """
    Обработчик сообщений от OpenAI Realtime API.
    Транслирует сообщения клиенту и обрабатывает специальные события.
    """
    if not openai_client.is_connected or not openai_client.ws:
        logger.error("OpenAI клиент не подключен")
        return
    
    try:
        logger.info(f"Начало обработки сообщений от OpenAI для клиента {openai_client.client_id}")
        
        while True:
            try:
                # Получаем сообщение от OpenAI
                raw_message = await openai_client.ws.recv()
                
                try:
                    response_data = json.loads(raw_message)
                except json.JSONDecodeError:
                    logger.error(f"Ошибка декодирования JSON от OpenAI: {raw_message[:200]}")
                    continue
                
                msg_type = response_data.get("type", "unknown")
                logger.info(f"Получено сообщение от OpenAI: тип={msg_type}")
                
                # Обработка различных типов сообщений
                if msg_type == "error":
                    # Ошибки от OpenAI
                    logger.error(f"Ошибка от OpenAI: {response_data}")
                    await websocket.send_json(response_data)
                    
                elif msg_type == "audio":
                    # Аудио данные от OpenAI - отправляем как бинарные данные
                    audio_base64 = response_data.get("data", "")
                    if audio_base64:
                        audio_chunk = base64.b64decode(audio_base64)
                        await websocket.send_bytes(audio_chunk)
                    
                elif msg_type == "conversation.item.input_audio_transcription.completed":
                    # Транскрипция речи пользователя
                    transcript = response_data.get("transcript", "")
                    logger.info(f"Пользователь сказал: '{transcript}'")
                    # Отправляем транскрипцию клиенту
                    await websocket.send_json({
                        "type": "user_transcript",
                        "transcript": transcript
                    })
                    
                elif msg_type == "response.audio_transcript.done":
                    # Транскрипция ответа ассистента
                    transcript = response_data.get("transcript", "")
                    logger.info(f"Ассистент ответил: '{transcript}'")
                    # Отправляем транскрипцию клиенту
                    await websocket.send_json({
                        "type": "assistant_transcript", 
                        "transcript": transcript
                    })
                    
                elif msg_type == "response.done":
                    # Завершение ответа
                    logger.info("Ответ ассистента завершен")
                    await websocket.send_json({
                        "type": "response_completed"
                    })
                    
                else:
                    # Все остальные сообщения транслируем как есть
                    await websocket.send_json(response_data)
                    
            except ConnectionClosed:
                logger.warning("Соединение с OpenAI закрыто")
                # Пытаемся переподключиться
                if await openai_client.reconnect():
                    logger.info("Соединение с OpenAI восстановлено")
                    continue
                else:
                    logger.error("Не удалось восстановить соединение с OpenAI")
                    await websocket.send_json({
                        "type": "error",
                        "error": {
                            "code": "openai_connection_lost",
                            "message": "Connection to AI lost"
                        }
                    })
                    break
                    
    except (ConnectionClosed, asyncio.CancelledError):
        logger.info(f"Обработка сообщений OpenAI завершена для клиента {openai_client.client_id}")
    except Exception as e:
        logger.error(f"Ошибка в обработчике сообщений OpenAI: {e}")
