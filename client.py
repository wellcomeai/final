import asyncio
import json
import uuid
import base64
import time
import websockets
import re
import logging
from websockets.exceptions import ConnectionClosed
from typing import Optional, List, Dict, Any, Union, AsyncGenerator

from config import settings

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "alloy"
DEFAULT_SYSTEM_MESSAGE = "Ты полезный голосовой ассистент. Отвечай кратко и по делу на русском языке."

def normalize_function_name(name: str) -> Optional[str]:
    """Нормализует имя функции к стандартному формату"""
    if not name:
        return None
    
    # Убираем пробелы и приводим к нижнему регистру
    normalized = name.strip().lower()
    
    # Заменяем пробелы и дефисы на подчеркивания
    normalized = re.sub(r'[-\s]+', '_', normalized)
    
    # Убираем все символы кроме букв, цифр и подчеркиваний
    normalized = re.sub(r'[^a-zA-Z0-9_]', '', normalized)
    
    return normalized if normalized else None

def normalize_functions(assistant_functions):
    """
    Преобразует список функций из UI в полные определения с параметрами.
    """
    if not assistant_functions:
        return []
    
    # Извлекаем имена функций
    enabled_names = []
    
    # Обработка формата {"enabled_functions": [...]}
    if isinstance(assistant_functions, dict) and "enabled_functions" in assistant_functions:
        enabled_names = [normalize_function_name(name) for name in assistant_functions.get("enabled_functions", [])]
    # Обработка списка объектов из UI
    else:
        enabled_names = [normalize_function_name(func.get("name")) for func in assistant_functions if func.get("name")]
    
    # Базовые определения функций
    function_definitions = {
        "send_webhook": {
            "name": "send_webhook",
            "description": "Отправляет данные на webhook URL для интеграции с внешними системами",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL вебхука для отправки данных"
                    },
                    "event": {
                        "type": "string", 
                        "description": "Тип события или действия"
                    },
                    "data": {
                        "type": "object",
                        "description": "Данные для отправки"
                    }
                },
                "required": ["url", "event"]
            }
        }
    }
    
    # Возвращаем определения для включенных функций
    result = []
    for name in enabled_names:
        if name and name in function_definitions:
            result.append(function_definitions[name])
    
    return result

def extract_webhook_url_from_prompt(prompt: str) -> Optional[str]:
    """Извлекает URL вебхука из системного промпта ассистента."""
    if not prompt:
        return None
        
    # Ищем URL с помощью регулярного выражения
    pattern1 = r'URL\s+(?:вебхука|webhook):\s*(https?://[^\s"\'<>]+)'
    pattern2 = r'(?:вебхука|webhook)\s+URL:\s*(https?://[^\s"\'<>]+)'
    pattern3 = r'https?://[^\s"\'<>]+'
    
    for pattern in [pattern1, pattern2, pattern3]:
        matches = re.findall(pattern, prompt, re.IGNORECASE)
        if matches:
            return matches[0]
            
    return None

def generate_short_id(prefix: str = "") -> str:
    """Генерирует короткий уникальный идентификатор длиной до 32 символов."""
    raw_id = str(uuid.uuid4()).replace("-", "")
    max_id_len = 32 - len(prefix)
    return f"{prefix}{raw_id[:max_id_len]}"

async def execute_function(name: str, arguments: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Выполняет функцию по имени с переданными аргументами.
    """
    try:
        if name == "send_webhook":
            # Импортируем httpx для отправки HTTP запросов
            import httpx
            
            url = arguments.get("url")
            event = arguments.get("event", "unknown")
            data = arguments.get("data", {})
            
            if not url:
                return {
                    "error": "URL not provided",
                    "status": "error"
                }
            
            # Подготавливаем данные для отправки
            payload = {
                "event": event,
                "data": data,
                "timestamp": time.time()
            }
            
            # Отправляем POST запрос
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        url,
                        json=payload,
                        timeout=10.0
                    )
                    
                    return {
                        "status": response.status_code,
                        "success": response.status_code < 400,
                        "response": response.text[:500],  # Ограничиваем размер ответа
                        "message": f"Webhook sent successfully with status {response.status_code}"
                    }
                except httpx.RequestError as e:
                    return {
                        "error": str(e),
                        "status": "error",
                        "success": False
                    }
        else:
            return {
                "error": f"Unknown function: {name}",
                "status": "error"
            }
            
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

class OpenAIRealtimeClient:
    """
    Клиент для взаимодействия с OpenAI Realtime API через WebSockets.
    Обрабатывает голосовые взаимодействия, вызовы функций и отслеживание разговоров.
    """
    
    def __init__(self, api_key: str, assistant_config: Dict[str, Any], client_id: str):
        """
        Инициализация клиента OpenAI Realtime.
        
        Args:
            api_key: OpenAI API ключ
            assistant_config: Конфигурация ассистента
            client_id: Уникальный идентификатор клиента
        """
        self.api_key = api_key
        self.assistant_config = assistant_config
        self.client_id = client_id
        self.ws = None
        self.is_connected = False
        self.openai_url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        self.session_id = str(uuid.uuid4())
        self.current_response_id = None
        self.webhook_url = None
        self.last_function_name = None
        self.enabled_functions = []
        
        # Извлекаем список разрешенных функций
        functions = assistant_config.get("functions", [])
        if isinstance(functions, list):
            self.enabled_functions = [normalize_function_name(f.get("name")) for f in functions if f.get("name")]
        elif isinstance(functions, dict) and "enabled_functions" in functions:
            self.enabled_functions = [normalize_function_name(name) for name in functions.get("enabled_functions", [])]
        
        logger.info(f"Извлечены разрешенные функции: {self.enabled_functions}")
        
        # Извлекаем URL вебхука из промпта только если функция send_webhook разрешена
        if "send_webhook" in self.enabled_functions:
            system_prompt = assistant_config.get("system_prompt", "")
            self.webhook_url = extract_webhook_url_from_prompt(system_prompt)
            if self.webhook_url:
                logger.info(f"Извлечен URL вебхука из промпта: {self.webhook_url}")

    async def connect(self) -> bool:
        """
        Устанавливает WebSocket соединение с OpenAI Realtime API
        и сразу отправляет актуальные настройки сессии.
        
        Returns:
            bool: True если соединение успешно, False иначе
        """
        if not self.api_key:
            logger.error("OpenAI API key не предоставлен")
            return False

        headers = [
            ("Authorization", f"Bearer {self.api_key}"),
            ("OpenAI-Beta", "realtime=v1"),
            ("User-Agent", "WellcomeAI/1.0")
        ]
        
        try:
            logger.info(f"Попытка подключения к OpenAI Realtime API: {self.openai_url}")
            self.ws = await asyncio.wait_for(
                websockets.connect(
                    self.openai_url,
                    extra_headers=headers,
                    max_size=settings.MAX_MESSAGE_SIZE,
                    ping_interval=settings.WEBSOCKET_PING_INTERVAL,
                    ping_timeout=settings.WEBSOCKET_PING_TIMEOUT,
                    close_timeout=settings.WEBSOCKET_CLOSE_TIMEOUT
                ),
                timeout=settings.CONNECTION_TIMEOUT
            )
            self.is_connected = True
            logger.info(f"Подключено к OpenAI для клиента {self.client_id}")

            # Получаем свежие настройки из assistant_config
            voice = self.assistant_config.get("voice", DEFAULT_VOICE)
            system_message = self.assistant_config.get("system_prompt", DEFAULT_SYSTEM_MESSAGE)
            functions = self.assistant_config.get("functions", [])
            
            # Обновляем список разрешенных функций
            if functions:
                if isinstance(functions, list):
                    self.enabled_functions = [normalize_function_name(f.get("name")) for f in functions if f.get("name")]
                elif isinstance(functions, dict) and "enabled_functions" in functions:
                    self.enabled_functions = [normalize_function_name(name) for name in functions.get("enabled_functions", [])]
                
                logger.info(f"Обновлены разрешенные функции: {self.enabled_functions}")

            # Проверяем URL вебхука в промпте, только если функция send_webhook разрешена
            if "send_webhook" in self.enabled_functions:
                self.webhook_url = extract_webhook_url_from_prompt(system_message)
                if self.webhook_url:
                    logger.info(f"Извлечен URL вебхука из промпта: {self.webhook_url}")

            # Отправляем обновленные настройки сессии
            if not await self.update_session(
                voice=voice,
                system_message=system_message,
                functions=functions
            ):
                logger.error("Не удалось обновить настройки сессии")
                await self.close()
                return False

            return True
        except asyncio.TimeoutError:
            logger.error(f"Таймаут подключения к OpenAI для клиента {self.client_id}")
            return False
        except Exception as e:
            logger.error(f"Ошибка подключения к OpenAI: {e}")
            return False

    async def reconnect(self) -> bool:
        """
        Пытается переподключиться к OpenAI Realtime API после потери соединения.
        
        Returns:
            bool: True если переподключение успешно, False иначе
        """
        logger.info(f"Попытка переподключения к OpenAI для клиента {self.client_id}")
        try:
            # Закрываем старое соединение, если оно ещё существует
            if self.ws:
                try:
                    await self.ws.close()
                except:
                    pass
            
            self.is_connected = False
            self.ws = None
            
            # Подключаемся заново
            return await self.connect()
        except Exception as e:
            logger.error(f"Ошибка при переподключении к OpenAI: {e}")
            return False

    async def update_session(
        self,
        voice: str = DEFAULT_VOICE,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        functions: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    ) -> bool:
        """
        Обновляет настройки сессии на стороне OpenAI Realtime API.
        
        Args:
            voice: Голос для синтеза речи
            system_message: Системные инструкции для ассистента
            functions: Список функций или словарь с enabled_functions
            
        Returns:
            bool: True если обновление успешно, False иначе
        """
        if not self.is_connected or not self.ws:
            logger.error("Невозможно обновить сессию: нет подключения")
            return False
            
        turn_detection = {
            "type": "server_vad",
            "threshold": 0.25,
            "prefix_padding_ms": 200,
            "silence_duration_ms": 300,
            "create_response": True  # ВАЖНО: автоматически создавать ответ после речи
        }
        
        # Получаем нормализованные определения функций
        normalized_functions = normalize_functions(functions)
        
        # Формируем tools для API
        tools = []
        for func_def in normalized_functions:
            tools.append({
                "type": "function",
                "name": func_def["name"],
                "description": func_def["description"],
                "parameters": func_def["parameters"]
            })
        
        # Обновляем список разрешенных функций на основе tools
        self.enabled_functions = [normalize_function_name(tool["name"]) for tool in tools]
        logger.info(f"Активированные функции для сессии: {self.enabled_functions}")
        
        # Устанавливаем tool_choice на основе наличия tools
        tool_choice = "auto" if tools else "none"
        
        logger.info(f"Настройка сессии с {len(tools)} функциями, tool_choice={tool_choice}")
        
        # Включение транскрипции аудио
        input_audio_transcription = {
            "model": "whisper-1"
        }
            
        payload = {
            "type": "session.update",
            "session": {
                "turn_detection": turn_detection,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "voice": voice,
                "instructions": system_message,
                "modalities": ["text", "audio"],  # ВАЖНО: оба режима
                "temperature": 0.8,  # Повышаем температуру для более естественной речи
                "max_response_output_tokens": 500,
                "tools": tools,
                "tool_choice": tool_choice,
                "input_audio_transcription": input_audio_transcription
            }
        }
        
        try:
            await self.ws.send(json.dumps(payload))
            logger.info(f"Настройки сессии отправлены (voice={voice}, tools={len(tools)}, tool_choice={tool_choice})")
            
            # Вывод подробной информации о функциях в лог
            if tools:
                for tool in tools:
                    logger.debug(f"Включена функция: {tool['name']}")
        except Exception as e:
            logger.error(f"Ошибка отправки session.update: {e}")
            return False

        return True

    async def send_function_result(self, function_call_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Отправляет результат выполнения функции обратно в OpenAI как событие conversation.item.create.
        
        Args:
            function_call_id: ID вызова функции
            result: Результат выполнения функции
            
        Returns:
            Dict: Информация о статусе доставки результата
        """
        if not self.is_connected or not self.ws:
            error_msg = "Невозможно отправить результат функции: нет подключения"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "payload": None
            }
        
        try:
            logger.info(f"Начало отправки результата функции: {function_call_id}")
            
            # Генерируем короткий ID длиной до 32 символов
            short_item_id = generate_short_id("func_")
            
            # Преобразуем результат в строку JSON
            result_json = json.dumps(result)
            
            # Структура для отправки результата функции
            payload = {
                "type": "conversation.item.create",
                "event_id": f"funcres_{time.time()}",
                "item": {
                    "id": short_item_id,
                    "type": "function_call_output",
                    "call_id": function_call_id,
                    "output": result_json
                }
            }
            
            logger.info(f"Отправка результата функции: {function_call_id}")
            
            await self.ws.send(json.dumps(payload))
            logger.info(f"Результат функции отправлен: {function_call_id}")
            
            # Добавляем небольшую задержку перед запросом нового ответа
            await asyncio.sleep(0.5)
            
            # После отправки результата, запрашиваем новый ответ от модели
            await self.create_response_after_function()
            
            logger.info(f"Результат функции отправлен и запрос на новый ответ выполнен")
            
            return {
                "success": True,
                "error": None,
                "payload": payload
            }
            
        except Exception as e:
            error_msg = f"Ошибка отправки результата функции: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "payload": None
            }

    async def create_response_after_function(self) -> bool:
        """
        Запрашивает новый ответ от модели после выполнения функции.
        Это обеспечит генерацию аудио-ответа.
        
        Returns:
            bool: True если успешно, False иначе
        """
        if not self.is_connected or not self.ws:
            logger.error("Невозможно создать ответ: нет подключения")
            return False
            
        try:
            logger.info(f"Создание нового ответа после выполнения функции")
            
            # Запрашиваем новый ответ от модели
            response_payload = {
                "type": "response.create",
                "event_id": f"resp_after_func_{time.time()}",
                "response": {
                    "modalities": ["text", "audio"],
                    "voice": self.assistant_config.get("voice", DEFAULT_VOICE),
                    "instructions": self.assistant_config.get("system_prompt", DEFAULT_SYSTEM_MESSAGE),
                    "temperature": 0.7,
                    "max_output_tokens": 200
                }
            }
            
            await self.ws.send(json.dumps(response_payload))
            logger.info("Запрошен новый ответ после выполнения функции")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка создания ответа после функции: {e}")
            return False

    async def process_audio(self, audio_buffer: bytes) -> bool:
        """Обрабатывает и отправляет аудио данные в OpenAI API"""
        if not self.is_connected or not self.ws or not audio_buffer:
            return False
        try:
            data_b64 = base64.b64encode(audio_buffer).decode("utf-8")
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": data_b64,
                "event_id": f"audio_{time.time()}"
            }))
            return True
        except ConnectionClosed:
            logger.warning("Соединение закрыто при отправке аудио данных")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Ошибка обработки аудио: {e}")
            return False

    async def commit_audio(self) -> bool:
        """Фиксирует аудио буфер, указывая, что пользователь закончил говорить"""
        if not self.is_connected or not self.ws:
            return False
        try:
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.commit",
                "event_id": f"commit_{time.time()}"
            }))
            return True
        except ConnectionClosed:
            logger.warning("Соединение закрыто при фиксации аудио")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Ошибка фиксации аудио: {e}")
            return False

    async def clear_audio_buffer(self) -> bool:
        """Очищает аудио буфер, удаляя все ожидающие аудио данные"""
        if not self.is_connected or not self.ws:
            return False
        try:
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.clear",
                "event_id": f"clear_{time.time()}"
            }))
            return True
        except ConnectionClosed:
            logger.warning("Соединение закрыто при очистке аудио буфера")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Ошибка очистки аудио буфера: {e}")
            return False

    async def recv(self):
        """Получает сообщение от WebSocket (совместимость с aiohttp версией)"""
        if not self.ws:
            raise Exception("WebSocket не подключен")
        
        return await self.ws.recv()

    async def close(self) -> None:
        """Закрывает WebSocket соединение"""
        if self.ws:
            try:
                await self.ws.close()
                logger.info(f"WebSocket соединение закрыто для клиента {self.client_id}")
            except Exception as e:
                logger.error(f"Ошибка закрытия WebSocket OpenAI: {e}")
        self.is_connected = False
