import base64
import struct
import numpy as np
import logging
from typing import Union

logger = logging.getLogger(__name__)

def base64_to_audio_buffer(base64_str: str) -> bytes:
    """
    Конвертация base64 строки в аудио буфер.
    
    Args:
        base64_str: Base64 закодированная строка
        
    Returns:
        bytes: Аудио данные как bytes
    """
    try:
        return base64.b64decode(base64_str)
    except Exception as e:
        logger.error(f"Ошибка конвертации base64 в аудио буфер: {e}")
        raise

def audio_buffer_to_base64(buffer: Union[bytes, bytearray, memoryview]) -> str:
    """
    Конвертация аудио буфера в base64 строку.
    
    Args:
        buffer: Аудио буфер как bytes, bytearray или memoryview
        
    Returns:
        str: Base64 закодированная строка
    """
    try:
        if isinstance(buffer, memoryview):
            buffer = buffer.tobytes()
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Ошибка конвертации аудио буфера в base64: {e}")
        raise

def create_wav_header(
    data_size: int,
    sample_rate: int = 24000,
    sample_width: int = 2,
    channels: int = 1
) -> bytes:
    """
    Создание WAV заголовка для PCM данных.
    
    Args:
        data_size: Размер аудио данных в байтах
        sample_rate: Частота дискретизации в Гц (по умолчанию 24000)
        sample_width: Ширина сэмпла в байтах (по умолчанию 2 для 16-bit)
        channels: Количество каналов (по умолчанию 1 для моно)
        
    Returns:
        bytes: WAV заголовок
    """
    try:
        # Вычисляем размеры
        file_size = 36 + data_size
        
        # Создаем заголовок
        header = bytearray()
        
        # RIFF chunk descriptor
        header.extend(b'RIFF')
        header.extend(struct.pack('<I', file_size))
        header.extend(b'WAVE')
        
        # fmt sub-chunk
        header.extend(b'fmt ')
        header.extend(struct.pack('<I', 16))  # Размер субчанка (16 для PCM)
        header.extend(struct.pack('<H', 1))   # Аудио формат (1 для PCM)
        header.extend(struct.pack('<H', channels))  # Количество каналов
        header.extend(struct.pack('<I', sample_rate))  # Частота дискретизации
        header.extend(struct.pack('<I', sample_rate * channels * sample_width))  # Байт-рейт
        header.extend(struct.pack('<H', channels * sample_width))  # Выравнивание блока
        header.extend(struct.pack('<H', sample_width * 8))  # Биты на сэмпл
        
        # data sub-chunk
        header.extend(b'data')
        header.extend(struct.pack('<I', data_size))
        
        return bytes(header)
        
    except Exception as e:
        logger.error(f"Ошибка создания WAV заголовка: {e}")
        raise

def create_wav_from_pcm(
    pcm_data: bytes,
    sample_rate: int = 24000,
    sample_width: int = 2,
    channels: int = 1
) -> bytes:
    """
    Создание WAV файла из PCM аудио данных.
    
    Args:
        pcm_data: PCM аудио данные
        sample_rate: Частота дискретизации в Гц (по умолчанию 24000)
        sample_width: Ширина сэмпла в байтах (по умолчанию 2 для 16-bit)
        channels: Количество каналов (по умолчанию 1 для моно)
        
    Returns:
        bytes: Полный WAV файл
    """
    try:
        header = create_wav_header(len(pcm_data), sample_rate, sample_width, channels)
        return header + pcm_data
    except Exception as e:
        logger.error(f"Ошибка создания WAV из PCM: {e}")
        raise

def float32_to_int16(float32_array: np.ndarray) -> np.ndarray:
    """
    Конвертация float32 numpy массива в int16.
    
    Args:
        float32_array: Float32 numpy массив в диапазоне [-1.0, 1.0]
        
    Returns:
        np.ndarray: Int16 numpy массив
    """
    try:
        # Убеждаемся что это float32 массив
        float32_array = np.asarray(float32_array, dtype=np.float32)
        
        # Обрезаем до диапазона [-1.0, 1.0] чтобы избежать переполнения
        float32_array = np.clip(float32_array, -1.0, 1.0)
        
        # Масштабируем до int16 диапазона и конвертируем
        return (float32_array * 32767.0).astype(np.int16)
    except Exception as e:
        logger.error(f"Ошибка конвертации float32 в int16: {e}")
        raise

def int16_to_float32(int16_array: np.ndarray) -> np.ndarray:
    """
    Конвертация int16 numpy массива в float32.
    
    Args:
        int16_array: Int16 numpy массив
        
    Returns:
        np.ndarray: Float32 numpy массив в диапазоне [-1.0, 1.0]
    """
    try:
        # Убеждаемся что это int16 массив
        int16_array = np.asarray(int16_array, dtype=np.int16)
        
        # Масштабируем до диапазона [-1.0, 1.0]
        return int16_array.astype(np.float32) / 32767.0
    except Exception as e:
        logger.error(f"Ошибка конвертации int16 в float32: {e}")
        raise
