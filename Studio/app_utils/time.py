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

from datetime import datetime


def get_timestamp_filename(extension: str = "mp4") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return f"{timestamp}.{extension}"


def format_duration(seconds: float) -> str:
    if seconds < 0:
        return "0s"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return " ".join(parts)


def parse_timestamp_filename(filename: str) -> datetime | None:
    try:
        name = filename.rsplit(".", 1)[0]
        return datetime.strptime(name, "%Y-%m-%d-%H-%M-%S")
    except (ValueError, IndexError):
        return None
