import base64
import struct
import numpy as np
import logging
from typing import Union

logger = logging.getLogger(__name__)

def audio_buffer_to_base64(buffer: Union[bytes, bytearray, memoryview]) -> str:
    """Конвертирует аудио буфер в base64 строку"""
    try:
        if isinstance(buffer, memoryview):
            buffer = buffer.tobytes()
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.exception(f"Ошибка конвертации аудио в base64: {str(e)}")
        raise

def base64_to_audio_buffer(base64_str: str) -> bytes:
    """Конвертирует base64 строку в аудио буфер"""
    try:
        return base64.b64decode(base64_str)
    except Exception as e:
        logger.exception(f"Ошибка конвертации base64 в аудио: {str(e)}")
        raise

def create_wav_from_pcm(
    pcm_data: bytes, 
    sample_rate: int = 24000, 
    sample_width: int = 2, 
    channels: int = 1
) -> bytes:
    """Создает WAV файл из PCM аудио данных"""
    try:
        data_size = len(pcm_data)
        file_size = 36 + data_size
        
        header = bytearray()
        
        # RIFF chunk descriptor
        header.extend(b'RIFF')
        header.extend(struct.pack('<I', file_size))
        header.extend(b'WAVE')
        
        # fmt sub-chunk
        header.extend(b'fmt ')
        header.extend(struct.pack('<I', 16))
        header.extend(struct.pack('<H', 1))
        header.extend(struct.pack('<H', channels))
        header.extend(struct.pack('<I', sample_rate))
        header.extend(struct.pack('<I', sample_rate * channels * sample_width))
        header.extend(struct.pack('<H', channels * sample_width))
        header.extend(struct.pack('<H', sample_width * 8))
        
        # data sub-chunk
        header.extend(b'data')
        header.extend(struct.pack('<I', data_size))
        
        wav_data = header + pcm_data
        
        return wav_data
    except Exception as e:
        logger.exception(f"Ошибка создания WAV из PCM: {str(e)}")
        raise
