from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio
import uuid
import base64
import traceback
from websockets.exceptions import ConnectionClosed
from typing import Dict, List

try:
    from client import OpenAIRealtimeClient, normalize_function_name, execute_function
except ImportError:
    # Fallback to aiohttp version if websockets doesn't work
    from client_aiohttp import OpenAIRealtimeClient, normalize_function_name, execute_function
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

        # Создаем конфигурацию ассистента
        assistant_config = {
            "name": "Voice Assistant",
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
        openai_client = OpenAIRealtimeClient(api_key, assistant_config, client_id)
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
                            # Пробуем восстановить соединение
                            if await openai_client.reconnect():
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
                            await openai_client.ws.send(json.dumps({
                                "type": "response.cancel",
                                "event_id": data.get("event_id")
                            }))
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
            await asyncio.sleep(0)

    finally:
        if openai_client:
            await openai_client.close()
        # убираем из active_connections
        conns = active_connections.get(assistant_id, [])
        if websocket in conns:
            conns.remove(websocket)
        print(f"Удалено WebSocket соединение: client_id={client_id}")


async def handle_openai_messages(openai_client: OpenAIRealtimeClient, websocket: WebSocket):
    """Обрабатывает сообщения от OpenAI и передает их клиенту"""
    if not openai_client.is_connected or not openai_client.ws:
        print("OpenAI клиент не подключен.")
        return
    
    # Переменные для хранения текста диалога и результата функции
    user_transcript = ""
    assistant_transcript = ""
    function_result = None
    
    # Буфер для накопления аргументов функции
    pending_function_call = {
        "name": None,
        "call_id": None,
        "arguments_buffer": ""
    }
    
    # Флаг ожидания ответа после вызова функции
    waiting_for_function_response = False
    last_function_delivery_status = None
    
    try:
        print(f"Начало обработки сообщений от OpenAI для клиента {openai_client.client_id}")
        print(f"Текущие разрешенные функции: {openai_client.enabled_functions}")
        
        while True:
            try:
                raw = await openai_client.recv()
                
                try:
                    response_data = json.loads(raw)
                except json.JSONDecodeError:
                    print(f"Ошибка декодирования JSON: {raw[:200]}")
                    continue
                    
                # Логирование типа полученного сообщения
                msg_type = response_data.get("type", "unknown")
                
                # Подробное логирование для ошибок
                if msg_type == "error":
                    print(f"ОШИБКА API: {json.dumps(response_data, ensure_ascii=False)}")
                    
                    # Если ошибка связана с отправкой результата функции
                    if waiting_for_function_response and "item" in str(response_data.get("error", {})):
                        error_message = response_data.get("error", {}).get("message", "Ошибка отправки результата функции")
                        print(f"Ошибка при отправке результата функции: {error_message}")
                        
                        # Создаем свое сообщение пользователю об ошибке
                        error_response = {
                            "type": "response.content_part.added",
                            "content": {
                                "text": f"Ошибка при выполнении функции: {error_message}"
                            }
                        }
                        await websocket.send_json(error_response)
                        
                        # Запрашиваем новый ответ для генерации аудио
                        await openai_client.create_response_after_function()
                        
                        # Сбрасываем флаг ожидания
                        waiting_for_function_response = False
                    else:
                        # Отправляем остальные ошибки клиенту
                        await websocket.send_json(response_data)
                    continue
                
                print(f"Получено сообщение от OpenAI: тип={msg_type}")
                
                # Обработка вызова функции
                if msg_type == "response.function_call.started":
                    function_name = response_data.get("function_name")
                    function_call_id = response_data.get("call_id")
                    
                    print(f"Начало вызова функции: {function_name}, ID: {function_call_id}")
                    
                    # Нормализуем имя функции
                    normalized_name = normalize_function_name(function_name) or function_name
                    
                    # Проверяем, разрешена ли функция
                    if normalized_name not in openai_client.enabled_functions:
                        print(f"Попытка вызвать неразрешенную функцию: {normalized_name}")
                        
                        # Отправляем сообщение об ошибке пользователю
                        error_response = {
                            "type": "response.content_part.added",
                            "content": {
                                "text": f"Ошибка: функция {function_name} не активирована для этого ассистента."
                            }
                        }
                        await websocket.send_json(error_response)
                        
                        # Отменяем вызов функции
                        if function_call_id:
                            dummy_result = {
                                "error": f"Функция {normalized_name} не разрешена",
                                "status": "error"
                            }
                            await openai_client.send_function_result(function_call_id, dummy_result)
                            
                        continue
                    
                    # Инициализируем данные о текущем вызове функции
                    pending_function_call = {
                        "name": normalized_name,
                        "call_id": function_call_id,
                        "arguments_buffer": ""
                    }
                    
                    # Уведомляем клиента о начале вызова функции
                    await websocket.send_json({
                        "type": "function_call.started",
                        "function": normalized_name,
                        "function_call_id": function_call_id
                    })
                
                # Обработка аргументов функции
                elif msg_type == "response.function_call_arguments.delta":
                    delta = response_data.get("delta", "")
                    
                    # Извлекаем имя функции и ID из первого delta, если их еще нет
                    if not pending_function_call["name"] and "call_id" in response_data:
                        pending_function_call["call_id"] = response_data.get("call_id")
                        
                        # Определение функции по содержимому аргументов
                        if "url" in delta or "event" in delta:
                            pending_function_call["name"] = "send_webhook"
                            print(f"Определена функция по аргументам: send_webhook")
                    
                    # Добавляем часть аргументов в буфер
                    pending_function_call["arguments_buffer"] += delta
                
                # Завершение получения аргументов и вызов функции
                elif msg_type == "response.function_call_arguments.done":
                    arguments_str = response_data.get("arguments", pending_function_call["arguments_buffer"])
                    
                    function_name = response_data.get("function_name", pending_function_call["name"])
                    function_call_id = response_data.get("call_id", pending_function_call["call_id"])
                    
                    print(f"Завершение получения аргументов для функции: {function_name}")
                    
                    # Восстановить имя функции по содержимому аргументов, если не определена
                    if not function_name and arguments_str:
                        if "url" in arguments_str:
                            function_name = "send_webhook"
                            print(f"Определена функция по аргументам: send_webhook")
                    
                    # Нормализация имени функции
                    normalized_name = normalize_function_name(function_name) or function_name
                    print(f"Нормализация окончательного имени: {function_name} -> {normalized_name}")
                    
                    # Проверяем, разрешена ли функция
                    if normalized_name and normalized_name not in openai_client.enabled_functions:
                        print(f"Попытка вызвать неразрешенную функцию: {normalized_name}")
                        
                        # Отправляем сообщение об ошибке пользователю
                        error_response = {
                            "type": "response.content_part.added",
                            "content": {
                                "text": f"Ошибка: функция {function_name} не активирована для этого ассистента."
                            }
                        }
                        await websocket.send_json(error_response)
                        
                        # Отправляем пустой результат, чтобы разблокировать модель
                        if function_call_id:
                            dummy_result = {
                                "error": f"Функция {normalized_name} не разрешена",
                                "status": "error"
                            }
                            await openai_client.send_function_result(function_call_id, dummy_result)
                            
                        # Сбрасываем буфер аргументов
                        pending_function_call = {
                            "name": None,
                            "call_id": None,
                            "arguments_buffer": ""
                        }
                        continue
                    
                    # Для разрешенных функций продолжаем нормальную обработку
                    if function_call_id and normalized_name:
                        print(f"Получены все аргументы функции {normalized_name}: {arguments_str}")
                        
                        try:
                            # Парсим аргументы из JSON-строки
                            arguments = json.loads(arguments_str)
                            
                            # Сообщаем клиенту о процессе выполнения функции
                            await websocket.send_json({
                                "type": "function_call.start",
                                "function": normalized_name,
                                "function_call_id": function_call_id
                            })
                            
                            # Выполняем функцию
                            result = await execute_function(
                                name=normalized_name,
                                arguments=arguments,
                                context={
                                    "assistant_config": openai_client.assistant_config,
                                    "client_id": openai_client.client_id
                                }
                            )
                            
                            # Сохраняем результат для логирования
                            function_result = result
                            
                            # Устанавливаем флаг ожидания ответа после вызова функции
                            waiting_for_function_response = True
                            
                            # Отправляем результат обратно в OpenAI
                            delivery_status = await openai_client.send_function_result(function_call_id, result)
                            last_function_delivery_status = delivery_status
                            
                            # Если произошла ошибка отправки результата
                            if not delivery_status["success"]:
                                print(f"Ошибка отправки результата функции: {delivery_status['error']}")
                                
                                # Генерируем сообщение для пользователя о проблеме
                                error_message = {
                                    "type": "response.content_part.added",
                                    "content": {
                                        "text": f"Произошла ошибка при выполнении функции: {delivery_status['error']}"
                                    }
                                }
                                await websocket.send_json(error_message)
                                
                                # Запрашиваем новый ответ для генерации аудио
                                await openai_client.create_response_after_function()
                                
                                # Сбрасываем флаг ожидания
                                waiting_for_function_response = False
                            
                            # Информируем клиента о результате в любом случае
                            await websocket.send_json({
                                "type": "function_call.completed",
                                "function": normalized_name,
                                "function_call_id": function_call_id,
                                "result": result
                            })
                            
                        except json.JSONDecodeError as e:
                            error_msg = f"Ошибка при парсинге аргументов функции: {e}"
                            print(error_msg)
                            await websocket.send_json({
                                "type": "error",
                                "error": {"code": "function_args_error", "message": error_msg}
                            })
                        except Exception as e:
                            error_msg = f"Ошибка при выполнении функции: {e}"
                            print(error_msg)
                            await websocket.send_json({
                                "type": "error",
                                "error": {"code": "function_execution_error", "message": error_msg}
                            })
                    
                    # Сбрасываем буфер аргументов для следующего вызова
                    pending_function_call = {
                        "name": None,
                        "call_id": None,
                        "arguments_buffer": ""
                    }

                # Обработка ответа с содержимым
                elif msg_type == "response.content_part.added":
                    # Если был вызов функции, и мы ждем ответа
                    if waiting_for_function_response:
                        print(f"Получен ответ после выполнения функции")
                        waiting_for_function_response = False
                    
                    # Обрабатываем текст ответа
                    if "text" in response_data.get("content", {}):
                        new_text = response_data.get("content", {}).get("text", "")
                        assistant_transcript = new_text
                        print(f"Получен текст ассистента: '{new_text}'")

                # Обработка транскрипции ввода пользователя
                if msg_type == "conversation.item.input_audio_transcription.completed":
                    if "transcript" in response_data:
                        user_transcript = response_data.get("transcript", "")
                        print(f"Получена транскрипция пользователя: '{user_transcript}'")

                # Обработка частей транскрипции для обоих типов сообщений
                if msg_type == "response.audio_transcript.delta":
                    delta_text = response_data.get("delta", "")
                    assistant_transcript += delta_text
                    print(f"Получен фрагмент транскрипции ассистента: '{delta_text}'")

                # Обработка полной транскрипции аудио ответа
                if msg_type == "response.audio_transcript.done":
                    transcript = response_data.get("transcript", "")
                    if transcript:
                        assistant_transcript = transcript
                        print(f"Получена полная транскрипция ассистента: '{assistant_transcript}'")

                # Если это аудио-чанк — отдаём как bytes
                if msg_type == "audio":
                    b64 = response_data.get("data", "")
                    chunk = base64.b64decode(b64)
                    await websocket.send_bytes(chunk)
                    continue

                # Завершение ответа
                if msg_type == "response.output_item.done":
                    # Если мы все еще ждем ответа функции и не получили контента
                    if waiting_for_function_response and last_function_delivery_status:
                        # Ассистент не обработал результат функции должным образом
                        if openai_client.last_function_name == "send_webhook" and function_result:
                            status_code = function_result.get("status", 0)
                            
                            # Генерируем информативный ответ в зависимости от статуса вебхука
                            message_text = ""
                            if status_code == 404:
                                message_text = "Вебхук не найден (ошибка 404). Возможно, он не зарегистрирован или не активирован."
                            elif status_code >= 200 and status_code < 300:
                                message_text = "Вебхук успешно выполнен."
                            else:
                                message_text = f"Вебхук вернул статус {status_code}."
                            
                            # Отправляем информацию клиенту
                            await websocket.send_json({
                                "type": "response.content_part.added",
                                "content": {
                                    "text": message_text
                                }
                            })
                            
                            # Запрашиваем новый ответ для генерации аудио
                            await openai_client.create_response_after_function()
                        
                        # Сбрасываем флаг ожидания
                        waiting_for_function_response = False

                # Все остальные сообщения отправляем как JSON
                await websocket.send_json(response_data)

                # Завершение диалога
                if msg_type == "response.done":
                    print(f"Получен сигнал завершения ответа")
                    
                    # Выводим финальные собранные тексты для анализа
                    print(f"Завершен диалог. Пользователь: '{user_transcript}'")
                    print(f"Завершен диалог. Ассистент: '{assistant_transcript}'")
                    
                    # Сбрасываем результат функции после логирования
                    function_result = None
                        
                    # Сбрасываем флаг ожидания функции, если он остался активным
                    waiting_for_function_response = False
                    
            except ConnectionClosed as e:
                print(f"Соединение с OpenAI закрыто: {e}")
                # Пробуем переподключиться
                if await openai_client.reconnect():
                    print("Соединение с OpenAI успешно восстановлено")
                    continue
                else:
                    print("Не удалось восстановить соединение с OpenAI")
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
        print(f"Трассировка: {traceback.format_exc()}")
