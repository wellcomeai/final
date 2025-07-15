"""
ElevenLabs Voice Chat - Python FastAPI Server
Полный аналог server-final.js с использованием FastAPI
"""

import os
import time
import json
import asyncio
import httpx
import psutil
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Конфигурация
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY', 'sk_95a5725ca01fdba20e15bd662d8b76152971016ff045377f')
AGENT_ID = os.getenv('AGENT_ID', 'agent_01jzwcew2ferttga9m1zcn3js1')
PORT = int(os.getenv('PORT', 10000))

app = FastAPI(
    title="ElevenLabs Voice Chat",
    description="Python FastAPI аналог Node.js сервера для ElevenLabs Conversational AI",
    version="2.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Логирование запросов
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    timestamp = datetime.now().isoformat()
    
    print(f"[{timestamp}] {request.method} {request.url.path} - {request.client.host}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    print(f"[{timestamp}] Response: {response.status_code} in {process_time:.3f}s")
    
    return response

print(f"🎯 Server starting with Agent ID: {AGENT_ID}")
print(f"🔑 API Key configured: {'Yes' if ELEVENLABS_API_KEY else 'No'}")

# ======================= ОСНОВНЫЕ API ENDPOINTS =======================

@app.get("/api/agent-id")
async def get_agent_id():
    """
    Основной endpoint - возвращает данные агента
    Аналог Node.js: app.get('/api/agent-id', ...)
    """
    print("📡 Agent ID requested")
    
    try:
        # Проверяем что агент существует
        agent_exists = await check_agent_exists()
        
        if agent_exists:
            return {
                "agent_id": AGENT_ID,
                "api_key": ELEVENLABS_API_KEY,
                "status": "ready",
                "source": "verified",
                "message": "Агент подтвержден и готов к работе",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Agent not found",
                    "status": "error",
                    "details": "Агент не найден в ElevenLabs",
                    "agent_id": AGENT_ID,
                    "timestamp": datetime.now().isoformat()
                }
            )
    except Exception as error:
        print(f"❌ Error checking agent: {error}")
        
        # Возвращаем данные без проверки если API недоступен
        return {
            "agent_id": AGENT_ID,
            "api_key": ELEVENLABS_API_KEY,
            "status": "ready",
            "source": "fallback",
            "message": "Агент готов (без проверки)",
            "warning": "Не удалось проверить статус агента в ElevenLabs",
            "timestamp": datetime.now().isoformat()
        }

async def check_agent_exists() -> bool:
    """
    Проверяет существование агента в ElevenLabs
    Аналог Node.js: function checkAgentExists()
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(
                f"https://api.elevenlabs.io/v1/convai/agents/{AGENT_ID}",
                headers={
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "User-Agent": "ElevenLabs-Voice-Chat/2.1",
                    "Accept": "application/json"
                }
            )
            
            print(f"📊 Agent check response: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ Agent exists and is accessible")
                return True
            elif response.status_code == 404:
                print("❌ Agent not found")
                return False
            elif response.status_code == 401:
                print("❌ Unauthorized - check API key")
                raise Exception("Unauthorized access to agent")
            else:
                print(f"⚠️ Unexpected status: {response.status_code}")
                raise Exception(f"API returned {response.status_code}")
                
        except httpx.TimeoutException:
            print("⏰ Agent check timeout")
            raise Exception("Request timeout")
        except Exception as e:
            print(f"❌ Agent check failed: {e}")
            raise

@app.get("/api/signed-url")
async def get_signed_url():
    """
    Получение signed URL для WebSocket подключения
    Аналог Node.js: app.get('/api/signed-url', ...)
    """
    print("🔐 Signed URL requested")
    
    try:
        # Сначала проверяем что агент существует
        print("Checking agent availability before signed URL...")
        agent_exists = await check_agent_exists()
        
        if not agent_exists:
            print("❌ Agent not found, cannot create signed URL")
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Agent not found",
                    "fallback_url": f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={AGENT_ID}",
                    "agent_id": AGENT_ID,
                    "details": "Agent does not exist or is not accessible",
                    "status": "agent_not_found",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        signed_url = await get_signed_url_from_api()
        print("✅ Signed URL obtained successfully")
        
        return {
            "signed_url": signed_url,
            "agent_id": AGENT_ID,
            "status": "ready",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as error:
        print(f"❌ Signed URL error: {error}")
        
        # Детальная обработка ошибок
        error_details = str(error)
        status_code = 500
        status = "error"
        
        if "Unauthorized" in error_details:
            status_code = 401
            status = "unauthorized"
            error_details = "Invalid API key or insufficient permissions"
        elif "Agent not found" in error_details:
            status_code = 404
            status = "agent_not_found"
            error_details = "Agent ID not found in ElevenLabs"
        elif "Rate limit" in error_details:
            status_code = 429
            status = "rate_limited"
            error_details = "API rate limit exceeded"
        elif "timeout" in error_details:
            status_code = 504
            status = "timeout"
            error_details = "ElevenLabs API timeout"
        
        return JSONResponse(
            status_code=status_code,
            content={
                "error": "Signed URL failed",
                "fallback_url": f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={AGENT_ID}",
                "agent_id": AGENT_ID,
                "details": error_details,
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "recommendations": get_error_recommendations(status)
            }
        )

async def get_signed_url_from_api() -> str:
    """
    Получает signed URL из ElevenLabs API
    Аналог Node.js: function getSignedUrl()
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get(
                f"https://api.elevenlabs.io/v1/convai/conversation/get-signed-url?agent_id={AGENT_ID}",
                headers={
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "User-Agent": "ElevenLabs-Voice-Chat/2.1",
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                }
            )
            
            print(f"📊 Signed URL response: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print("Signed URL response:", data)
                if "signed_url" in data:
                    return data["signed_url"]
                else:
                    raise Exception("No signed_url in response")
            elif response.status_code == 401:
                raise Exception("Unauthorized - check API key")
            elif response.status_code == 404:
                raise Exception("Agent not found or endpoint not found")
            elif response.status_code == 429:
                raise Exception("Rate limit exceeded")
            else:
                error_msg = f"API error: {response.status_code}"
                try:
                    error_data = response.json()
                    if "detail" in error_data:
                        error_msg += f" - {error_data['detail']}"
                    elif "error" in error_data:
                        error_msg += f" - {error_data['error']}"
                except:
                    error_msg += f" - {response.text}"
                print(f"Full error response: {response.text}")
                raise Exception(error_msg)
                
        except httpx.TimeoutException:
            print("⏰ Request timeout")
            raise Exception("Request timeout - ElevenLabs API not responding")
        except Exception as e:
            print(f"🌐 Network error: {e}")
            raise

def get_error_recommendations(status: str) -> List[str]:
    """Рекомендации по устранению ошибок"""
    recommendations = {
        "unauthorized": [
            "Check your ElevenLabs API key",
            "Verify API key has proper permissions",
            "Check if API key is expired"
        ],
        "agent_not_found": [
            "Verify agent ID in ElevenLabs Dashboard",
            "Check if agent is active and published",
            "Ensure agent is accessible with current API key"
        ],
        "rate_limited": [
            "Wait before retrying",
            "Check your ElevenLabs usage limits",
            "Consider upgrading your plan"
        ],
        "timeout": [
            "Check internet connection",
            "Retry after a few moments",
            "ElevenLabs API may be experiencing issues"
        ]
    }
    
    return recommendations.get(status, [
        "Try refreshing the page",
        "Check ElevenLabs service status",
        "Contact support if problem persists"
    ])

@app.get("/health")
async def health_check():
    """
    Health check с подробной диагностикой
    Аналог Node.js: app.get('/health', ...)
    """
    health = {
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "uptime": time.time() - psutil.boot_time(),
        "memory": dict(psutil.virtual_memory()._asdict()),
        "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
        "port": PORT,
        "agent_id": AGENT_ID,
        "api_configured": bool(ELEVENLABS_API_KEY)
    }

    try:
        # Быстрая проверка доступности ElevenLabs API
        await check_elevenlabs_api()
        health["elevenlabs_api"] = "accessible"
        health["agent_ready"] = True
    except Exception as error:
        health["elevenlabs_api"] = "error"
        health["agent_ready"] = False
        health["api_error"] = str(error)

    status_code = 200 if health["elevenlabs_api"] == "accessible" else 503
    return JSONResponse(status_code=status_code, content=health)

async def check_elevenlabs_api():
    """Быстрая проверка доступности ElevenLabs API"""
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(
            "https://api.elevenlabs.io/v1/user",
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "User-Agent": "ElevenLabs-Voice-Chat/2.1"
            }
        )
        
        if response.status_code not in [200, 401]:
            # 200 = OK, 401 = API key issue but API is accessible
            raise Exception(f"API status: {response.status_code}")

@app.post("/api/retry-agent")
async def retry_agent():
    """
    Retry agent endpoint
    Аналог Node.js: app.post('/api/retry-agent', ...)
    """
    print("🔄 Agent retry requested")
    
    try:
        exists = await check_agent_exists()
        
        if exists:
            return {
                "success": True,
                "agent_id": AGENT_ID,
                "status": "ready",
                "message": "Agent is ready"
            }
        else:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": "Agent not found",
                    "agent_id": AGENT_ID,
                    "message": "Agent does not exist in ElevenLabs"
                }
            )
    except Exception as error:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(error),
                "agent_id": AGENT_ID,
                "message": "Failed to verify agent"
            }
        )

@app.get("/api/diagnostics")
async def get_diagnostics():
    """
    Подробная диагностика системы
    Аналог Node.js: app.get('/api/diagnostics', ...)
    """
    print("🔍 Diagnostics requested")
    
    diagnostics = {
        "timestamp": datetime.now().isoformat(),
        "server": {
            "status": "running",
            "uptime": time.time() - psutil.boot_time(),
            "memory": dict(psutil.virtual_memory()._asdict()),
            "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
            "port": PORT
        },
        "configuration": {
            "agent_id": AGENT_ID,
            "api_key_configured": bool(ELEVENLABS_API_KEY),
            "api_key_preview": f"{ELEVENLABS_API_KEY[:8]}..." if ELEVENLABS_API_KEY else "not set"
        },
        "endpoints": {
            "health": "/health",
            "agent_id": "/api/agent-id",
            "signed_url": "/api/signed-url",
            "diagnostics": "/api/diagnostics"
        },
        "recommendations": [],
        "tests": {}
    }

    # Test 1: ElevenLabs API accessibility
    try:
        await check_elevenlabs_api()
        diagnostics["elevenlabs"] = {
            "status": "accessible",
            "message": "API is responding",
            "test_endpoint": "/v1/user"
        }
        diagnostics["recommendations"].append("✅ ElevenLabs API доступен")
        diagnostics["tests"]["api_connectivity"] = "passed"
    except Exception as error:
        diagnostics["elevenlabs"] = {
            "status": "error",
            "message": str(error),
            "test_endpoint": "/v1/user"
        }
        diagnostics["recommendations"].append("❌ Проблема с ElevenLabs API")
        diagnostics["recommendations"].append("💡 Проверьте API ключ и интернет-соединение")
        diagnostics["tests"]["api_connectivity"] = "failed"

    # Test 2: Agent existence and accessibility
    try:
        agent_exists = await check_agent_exists()
        if agent_exists:
            diagnostics["agent"] = {
                "status": "found",
                "id": AGENT_ID,
                "test_endpoint": f"/v1/convai/agents/{AGENT_ID}"
            }
            diagnostics["recommendations"].append("✅ Агент найден и доступен")
            diagnostics["tests"]["agent_accessibility"] = "passed"
        else:
            diagnostics["agent"] = {
                "status": "not_found",
                "id": AGENT_ID,
                "test_endpoint": f"/v1/convai/agents/{AGENT_ID}"
            }
            diagnostics["recommendations"].append("❌ Агент не найден")
            diagnostics["recommendations"].append("💡 Проверьте ID агента в ElevenLabs Dashboard")
            diagnostics["tests"]["agent_accessibility"] = "failed"
    except Exception as error:
        diagnostics["agent"] = {
            "status": "error",
            "error": str(error),
            "id": AGENT_ID
        }
        diagnostics["recommendations"].append("⚠️ Не удалось проверить статус агента")
        diagnostics["tests"]["agent_accessibility"] = "error"

    # Test 3: Signed URL generation
    try:
        signed_url = await get_signed_url_from_api()
        diagnostics["signed_url"] = {
            "status": "working",
            "message": "Can generate signed URLs",
            "url_preview": signed_url[:80] + "..."
        }
        diagnostics["recommendations"].append("✅ Signed URL генерация работает")
        diagnostics["tests"]["signed_url_generation"] = "passed"
    except Exception as error:
        diagnostics["signed_url"] = {
            "status": "error",
            "message": str(error)
        }
        diagnostics["recommendations"].append("⚠️ Проблема с генерацией Signed URL")
        diagnostics["recommendations"].append("💡 Будет использовано прямое подключение")
        diagnostics["tests"]["signed_url_generation"] = "failed"

    # Overall health assessment
    passed_tests = sum(1 for test in diagnostics["tests"].values() if test == "passed")
    total_tests = len(diagnostics["tests"])
    
    diagnostics["overall"] = {
        "health_score": f"{passed_tests}/{total_tests}",
        "status": "healthy" if passed_tests == total_tests else 
                 "partial" if passed_tests > 0 else "unhealthy",
        "ready_for_connection": passed_tests >= 1  # Минимум API должен быть доступен
    }

    # Additional recommendations based on overall health
    if diagnostics["overall"]["status"] == "unhealthy":
        diagnostics["recommendations"].append("🚨 Система не готова к работе")
        diagnostics["recommendations"].append("💡 Проверьте все настройки и попробуйте позже")
    elif diagnostics["overall"]["status"] == "partial":
        diagnostics["recommendations"].append("⚠️ Система частично готова")
        diagnostics["recommendations"].append("💡 Некоторые функции могут не работать")
    else:
        diagnostics["recommendations"].append("🎉 Система полностью готова к работе")

    return diagnostics

# ======================= СТАТИЧЕСКИЕ ФАЙЛЫ =======================

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Главная страница приложения"""
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>File not found</h1><p>templates/index.html not found</p>",
            status_code=404
        )

@app.get("/debug", response_class=HTMLResponse)
async def serve_debug():
    """Debug страница"""
    try:
        with open("templates/debug.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>File not found</h1><p>templates/debug.html not found</p>",
            status_code=404
        )

@app.get("/favicon.ico")
async def favicon():
    """Favicon - возвращаем пустой ответ"""
    return JSONResponse(status_code=204, content=None)

# ======================= ERROR HANDLING =======================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Глобальный обработчик ошибок"""
    print(f"❌ Server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """404 обработчик"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "path": str(request.url.path),
            "method": request.method,
            "timestamp": datetime.now().isoformat()
        }
    )

# ======================= STARTUP/SHUTDOWN =======================

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    print(f"🚀 Server running on port {PORT}")
    print(f"🎯 Agent ID: {AGENT_ID}")
    print(f"✅ All endpoints ready!")
    print(f"📱 App: http://localhost:{PORT}")
    print(f"🔧 Debug: http://localhost:{PORT}/debug")
    print(f"🩺 Health: http://localhost:{PORT}/health")
    
    # Начальная проверка здоровья
    async def initial_health_check():
        await asyncio.sleep(1)
        try:
            await check_elevenlabs_api()
            print("✅ Initial ElevenLabs API check passed")
        except Exception as error:
            print(f"⚠️ Initial ElevenLabs API check failed: {error}")
    
    # Запускаем проверку в фоне
    asyncio.create_task(initial_health_check())

@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown"""
    print("🛑 Shutting down gracefully...")

# ======================= MAIN =======================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,  # Для production на Render
        log_level="info"
    )
