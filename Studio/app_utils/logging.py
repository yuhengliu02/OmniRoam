'''
ADOBE CONFIDENTIAL
Copyright 2026 Adobe
All Rights Reserved.
NOTICE: All information contained herein is, and remains
the property of Adobe and its suppliers, if any. The intellectual
and technical concepts contained herein are proprietary to Adobe
and its suppliers and are protected by all applicable intellectual
property laws, including trade secret and copyright laws.
Dissemination of this information or reproduction of this material
is strictly forbidden unless prior written permission is obtained
from Adobe.
'''

from __future__ import annotations

import logging
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import List, Optional


class LogBuffer:
    
    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self._entries: List[str] = []
    
    def append(self, message: str, level: str = "INFO") -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        self._entries.append(entry)
        
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]
    
    def info(self, message: str) -> None:
        self.append(message, "INFO")
    
    def error(self, message: str) -> None:
        self.append(message, "ERROR")
    
    def warning(self, message: str) -> None:
        self.append(message, "WARN")
    
    def debug(self, message: str) -> None:
        self.append(message, "DEBUG")
    
    def get_all(self) -> str:
        return "\n".join(self._entries)
    
    def get_recent(self, n: int = 100) -> str:
        return "\n".join(self._entries[-n:])
    
    def clear(self) -> None:
        self._entries = []
    
    def save_to_file(self, path: Path) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(self.get_all())
            return True
        except Exception:
            return False


_log_buffer: Optional[LogBuffer] = None


def get_log_buffer() -> LogBuffer:
    global _log_buffer
    if _log_buffer is None:
        _log_buffer = LogBuffer()
    return _log_buffer


def get_logger(name: str = "omniroam") -> logging.Logger:
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        buffer_handler = LogBufferHandler(get_log_buffer())
        buffer_handler.setLevel(logging.DEBUG)
        logger.addHandler(buffer_handler)
    
    return logger


class LogBufferHandler(logging.Handler):
    
    def __init__(self, buffer: LogBuffer):
        super().__init__()
        self.buffer = buffer
    
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            level = record.levelname
            self.buffer.append(record.getMessage(), level)
        except Exception:
            self.handleError(record)
