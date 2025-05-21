import asyncio
import json
import uuid
import base64
import time
import httpx
import websockets
import socket
from websockets.exceptions import ConnectionClosed
from typing import Dict, Any, Optional, List, Tuple

from config import settings

class SimpleOpenAIClient:
    """Упрощенный клиент для работы с OpenAI Realtime API через WebSockets"""
    
    def __init__(self, api_key: Optional[str] = None, assistant_config: Optional[Dict] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.assistant_config = assistant_config or settings.DEFAULT_ASSISTANT_CONFIG
        self.client_id = str(uuid.uuid4())
        self.ws = None
        self.is_connected = False
        self.session_id = None
        self.current_response_id = None
        self.session_token = None
        
        # Простой список разрешенных функций
        self.enabled_functions = []
        if "functions" in self.assistant_config:
            self.enabled_functions = [f.get("name") for f in self.assistant_config["functions"] if f.get("name")]

    async def resolve_hostname(self, hostname: str) -> List[Tuple[str, str]]:
        """Вспомогательная функция для резолвинга IP-адресов по имени хоста"""
        try:
            # Получаем информацию о хосте
            result = []
            addr_info = socket.getaddrinfo(hostname, 443, family=socket.AF_INET, 
                                         type=socket.SOCK_STREAM)
            
            for family, socktype, proto, canonname, sockaddr in addr_info:
                ip, port = sockaddr
                result.append((ip, canonname or hostname))
                
            return result
        except socket.gaierror as e:
            print(f"Ошибка резолвинга {hostname}: {e}")
            return []

    async def create_session(self) -> bool:
        """Создает сессию в OpenAI Realtime API через REST запрос"""
        if not self.api_key:
            print("OpenAI API key не предоставлен")
            return False
            
        try:
            # Получаем настройки
            voice = self.assistant_config.get("voice", settings.DEFAULT_VOICE)
            system_message = self.assistant_config.get("system_prompt", settings.DEFAULT_SYSTEM_MESSAGE)
            model = self.assistant_config.get("model", settings.DEFAULT_MODEL)
            
            print(f"Создание сессии с моделью {model}, голосом {voice}")
            
            # Подготавливаем данные для запроса на создание сессии
            session_data = {
                "model": model,
                "modalities": ["audio", "text"],
                "instructions": system_message,
                "voice": voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.25,
                    "prefix_padding_ms": 200,
                    "silence_duration_ms": 300,
                    "create_response": True,
                }
            }
            
            # Добавляем функции, если они есть
            if self.assistant_config.get("functions"):
                tools = []
                for func in self.assistant_config["functions"]:
                    if "name" in func and "description" in func and "parameters" in func:
                        tools.append({
                            "type": "function",
                            "name": func["name"],
                            "description": func["description"],
                            "parameters": func["parameters"]
                        })
                
                if tools:
                    session_data["tools"] = tools
                    session_data["tool_choice"] = "auto"
            
            # Делаем запрос на создание сессии
            print(f"Отправка запроса на: {settings.REALTIME_SESSION_URL}")
            
            # Пробуем предварительно разрешить домен
            api_host = "api.openai.com"
            resolved_ips = await self.resolve_hostname(api_host)
            if resolved_ips:
                for ip, name in resolved_ips:
                    print(f"✓ Успешно разрешен {api_host} -> {ip}")
            else:
                print(f"⚠ Не удалось разрешить {api_host}")
            
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "OpenAI-Beta": "realtime=v1"
                }
                
                try:
                    response = await client.post(
                        settings.REALTIME_SESSION_URL,
                        headers=headers,
                        json=session_data,
                        timeout=30.0
                    )
                    
                    if response.status_code != 200:
                        print(f"Ошибка создания сессии: {response.status_code} {response.text}")
                        return False
                    
                    session_info = response.json()
                    if "client_secret" not in session_info or not session_info["client_secret"].get("value"):
                        print("Не удалось получить токен сессии")
                        return False
                    
                    self.session_token = session_info["client_secret"]["value"]
                    self.session_id = session_info["id"]
                    
                    print(f"Сессия создана успешно, получен токен: {self.session_token[:10]}...")
                    return True
                    
                except httpx.ConnectError as e:
                    print(f"Ошибка подключения к API OpenAI: {e}")
                    # Проверяем прямое подключение по IP, если домен разрешен
                    if resolved_ips:
                        ip = resolved_ips[0][0]
                        print(f"Пробуем подключиться напрямую к IP: {ip}")
                        try:
                            # Создаем URL с IP-адресом
                            direct_url = f"https://{ip}/v1/realtime/sessions"
                            headers["Host"] = api_host  # Важно добавить заголовок Host
                            
                            response = await client.post(
                                direct_url,
                                headers=headers,
                                json=session_data,
                                timeout=30.0
                            )
                            
                            if response.status_code != 200:
                                print(f"Ошибка создания сессии по IP: {response.status_code} {response.text}")
                                return False
                            
                            session_info = response.json()
                            if "client_secret" not in session_info or not session_info["client_secret"].get("value"):
                                print("Не удалось получить токен сессии (IP)")
                                return False
                            
                            self.session_token = session_info["client_secret"]["value"]
                            self.session_id = session_info["id"]
                            
                            print(f"Сессия создана успешно через прямой IP, получен токен: {self.session_token[:10]}...")
                            return True
                            
                        except Exception as inner_e:
                            print(f"Ошибка подключения по IP: {inner_e}")
                            return False
                    return False
                
        except Exception as e:
            print(f"Ошибка создания сессии OpenAI: {e}")
            return False

    async def connect(self) -> bool:
        """Устанавливает WebSocket соединение с OpenAI Realtime API"""
        if not self.api_key:
            print("OpenAI API key не предоставлен")
            return False

        try:
            # Сначала создаем сессию через REST API
            if not await self.create_session():
                print("Не удалось создать сессию")
                return False
                
            # Пробуем несколько вариантов URL для WebSocket соединения
            # Варианты URL для подключения
            ws_urls = [
                "wss://realtime.api.openai.com/v1/audio/speech",
                "wss://realtime.api.openai.com/v1/audio/speech/",
                "wss://realtime.api.openai.com/v1"
            ]
            
            # Пробуем разрешить домен WebSocket
            ws_host = "realtime.api.openai.com"
            resolved_ips = await self.resolve_hostname(ws_host)
            if resolved_ips:
                for ip, name in resolved_ips:
                    print(f"✓ Успешно разрешен {ws_host} -> {ip}")
                    # Добавляем URL с прямым IP
                    ws_urls.append(f"wss://{ip}/v1/audio/speech")
            else:
                print(f"⚠ Не удалось разрешить {ws_host}, используем только доменное имя")
            
            # Пробуем подключиться через proxy.hf.space (Hugging Face) для обхода ограничений Render
            ws_urls.append("wss://proxy.hf.space/proxy/openai/realtime/v1/audio/speech")
            
            # Перебираем все URL и пробуем подключиться
            last_error = None
            for ws_url in ws_urls:
                print(f"Попытка подключения к WebSocket: {ws_url}")
                
                # Подключаемся через WebSocket, используя полученный токен сессии
                headers = [
                    ("Authorization", f"Bearer {self.session_token}"),
                    ("OpenAI-Beta", "realtime=v1"),
                    ("User-Agent", "MiniVoiceAssistant/1.0")
                ]
                
                # Если используем прямой IP, добавляем заголовок Host
                if ws_url.replace("wss://", "").startswith(("192.", "10.", "172.")):
                    headers.append(("Host", ws_host))
                
                try:
                    self.ws = await asyncio.wait_for(
                        websockets.connect(
                            ws_url,
                            extra_headers=headers,
                            max_size=15*1024*1024,
                            ping_interval=30,
                            ping_timeout=120,
                            close_timeout=15
                        ),
                        timeout=10  # Уменьшаем таймаут для каждой попытки
                    )
                    self.is_connected = True
                    print(f"Подключено к OpenAI для клиента {self.client_id} через {ws_url}")
                    
                    # Ждем первого сообщения от сервера (должно быть session.created)
                    try:
                        first_message = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
                        print(f"Получено первое сообщение от сервера: {first_message[:100]}...")
                    except asyncio.TimeoutError:
                        print("Таймаут при ожидании первого сообщения от сервера")
                    except Exception as e:
                        print(f"Ошибка при получении первого сообщения: {e}")
                    
                    return True
                except Exception as e:
                    last_error = e
                    print(f"Не удалось подключиться через {ws_url}: {e}")
                    continue
            
            print(f"Все попытки подключения к WebSocket завершились неудачно. Последняя ошибка: {last_error}")
            return False
            
        except asyncio.TimeoutError:
            print(f"Таймаут подключения к OpenAI для клиента {self.client_id}")
            return False
        except Exception as e:
            print(f"Ошибка подключения к OpenAI: {e}")
            return False

    async def update_session(self, voice: str = None, system_message: str = None) -> bool:
        """Обновляет настройки сессии на стороне OpenAI Realtime API"""
        if not self.is_connected or not self.ws:
            print("Невозможно обновить сессию: нет подключения")
            return False
            
        voice = voice or settings.DEFAULT_VOICE
        system_message = system_message or settings.DEFAULT_SYSTEM_MESSAGE
            
        turn_detection = {
            "type": "server_vad",
            "threshold": 0.25,
            "prefix_padding_ms": 200,
            "silence_duration_ms": 300,
            "create_response": True,
        }
        
        # Упрощенные настройки без функций для минимальной версии
        payload = {
            "type": "session.update",
            "session": {
                "turn_detection": turn_detection,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "voice": voice,
                "instructions": system_message,
                "modalities": ["text", "audio"],
                "temperature": 0.7,
                "max_response_output_tokens": 500,
                "input_audio_transcription": {
                    "model": "whisper-1"
                }
            }
        }
        
        # Добавляем функции, если они есть
        if self.assistant_config.get("functions"):
            tools = []
            for func in self.assistant_config["functions"]:
                if "name" in func and "description" in func and "parameters" in func:
                    tools.append({
                        "type": "function",
                        "name": func["name"],
                        "description": func["description"],
                        "parameters": func["parameters"]
                    })
            
            if tools:
                payload["session"]["tools"] = tools
                payload["session"]["tool_choice"] = "auto"
                print(f"Добавлено {len(tools)} функций в сессию")
            
        try:
            await self.ws.send(json.dumps(payload))
            print(f"Настройки сессии отправлены (voice={voice})")
            return True
        except Exception as e:
            print(f"Ошибка отправки session.update: {e}")
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
            print("Соединение закрыто при отправке аудио данных")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"Ошибка обработки аудио: {e}")
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
            print("Соединение закрыто при фиксации аудио")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"Ошибка фиксации аудио: {e}")
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
            print("Соединение закрыто при очистке аудио буфера")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"Ошибка очистки аудио буфера: {e}")
            return False
            
    async def cancel_response(self) -> bool:
        """Отменяет текущий ответ ассистента"""
        if not self.is_connected or not self.ws or not self.current_response_id:
            return False
        try:
            await self.ws.send(json.dumps({
                "type": "response.cancel",
                "event_id": f"cancel_{time.time()}",
                "response_id": self.current_response_id
            }))
            print(f"Отправлена команда отмены ответа: {self.current_response_id}")
            return True
        except ConnectionClosed:
            print("Соединение закрыто при отмене ответа")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"Ошибка отмены ответа: {e}")
            return False

    async def close(self) -> None:
        """Закрывает WebSocket соединение"""
        if self.ws:
            try:
                await self.ws.close()
                print(f"WebSocket соединение закрыто для клиента {self.client_id}")
            except Exception as e:
                print(f"Ошибка закрытия WebSocket OpenAI: {e}")
        self.is_connected = False
