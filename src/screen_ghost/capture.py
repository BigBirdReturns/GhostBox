"""
Real Capture Sources for GhostBox v0.7

This module provides integration points for real data sources:
- File watchers (JSON, CSV, log files)
- SQLite databases (ScreenGhost logs, other tools)
- HTTP endpoints (webhooks, polling)
- Platform-specific captures (Android, desktop)

The goal: plug in your actual data without changing the stack.
"""

from __future__ import annotations

import json
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional
from queue import Queue
from threading import Thread

from .simulated import Event


class CaptureSource(ABC):
    """
    Abstract base for all capture sources.
    
    Implement this to connect any data stream to the GhostBox stack.
    """
    
    @abstractmethod
    def events(self) -> Iterator[Event]:
        """Yield events from this source."""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Human-readable source name."""
        pass


# =============================================================================
# File-Based Sources
# =============================================================================

@dataclass
class JSONFileSource(CaptureSource):
    """
    Read events from a JSON file or directory of JSON files.
    
    Expected format per file:
    [
      {"topic": "...", "source": "...", "text": "...", "confidence": 0.9},
      ...
    ]
    
    Or newline-delimited JSON (NDJSON).
    """
    
    path: Path
    source_name: str = "json_file"
    _consumed: set = field(default_factory=set)
    
    def name(self) -> str:
        return f"json:{self.path}"
    
    def events(self) -> Iterator[Event]:
        if self.path.is_file():
            yield from self._read_file(self.path)
        elif self.path.is_dir():
            for f in sorted(self.path.glob("*.json")):
                if str(f) not in self._consumed:
                    yield from self._read_file(f)
                    self._consumed.add(str(f))
    
    def _read_file(self, fp: Path) -> Iterator[Event]:
        text = fp.read_text()
        
        # Try array format first
        try:
            data = json.loads(text)
            if isinstance(data, list):
                for item in data:
                    yield self._to_event(item)
                return
        except json.JSONDecodeError:
            pass
        
        # Try NDJSON
        for line in text.strip().split("\n"):
            if line.strip():
                try:
                    item = json.loads(line)
                    yield self._to_event(item)
                except json.JSONDecodeError:
                    continue
    
    def _to_event(self, item: Dict[str, Any]) -> Event:
        return Event(
            topic=item.get("topic", "unknown"),
            source=item.get("source", self.source_name),
            text=item.get("text", ""),
            confidence=float(item.get("confidence", 0.5)),
        )


@dataclass
class CSVFileSource(CaptureSource):
    """
    Read events from a CSV file.
    
    Expected columns: topic, source, text, confidence
    """
    
    path: Path
    source_name: str = "csv_file"
    
    def name(self) -> str:
        return f"csv:{self.path}"
    
    def events(self) -> Iterator[Event]:
        import csv
        
        with open(self.path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield Event(
                    topic=row.get("topic", "unknown"),
                    source=row.get("source", self.source_name),
                    text=row.get("text", ""),
                    confidence=float(row.get("confidence", 0.5)),
                )


@dataclass
class LogFileSource(CaptureSource):
    """
    Read events from a log file with a custom parser.
    
    Provide a parser function that takes a line and returns an Event or None.
    """
    
    path: Path
    parser: Callable[[str], Optional[Event]]
    source_name: str = "log_file"
    _last_position: int = 0
    
    def name(self) -> str:
        return f"log:{self.path}"
    
    def events(self) -> Iterator[Event]:
        with open(self.path) as f:
            f.seek(self._last_position)
            for line in f:
                event = self.parser(line)
                if event is not None:
                    yield event
            self._last_position = f.tell()


# =============================================================================
# Database Sources
# =============================================================================

@dataclass
class SQLiteSource(CaptureSource):
    """
    Read events from a SQLite database.
    
    Designed to work with ScreenGhost's native log format,
    but configurable for any schema.
    """
    
    db_path: Path
    table: str = "events"
    topic_col: str = "topic"
    source_col: str = "source"
    text_col: str = "text"
    confidence_col: str = "confidence"
    timestamp_col: str = "created_at"
    source_name: str = "sqlite"
    _last_timestamp: Optional[str] = None
    
    def name(self) -> str:
        return f"sqlite:{self.db_path}:{self.table}"
    
    def events(self) -> Iterator[Event]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        query = f"""
            SELECT {self.topic_col}, {self.source_col}, {self.text_col}, 
                   {self.confidence_col}, {self.timestamp_col}
            FROM {self.table}
        """
        
        if self._last_timestamp:
            query += f" WHERE {self.timestamp_col} > ?"
            cursor = conn.execute(query, (self._last_timestamp,))
        else:
            cursor = conn.execute(query)
        
        for row in cursor:
            self._last_timestamp = row[self.timestamp_col]
            yield Event(
                topic=row[self.topic_col] or "unknown",
                source=row[self.source_col] or self.source_name,
                text=row[self.text_col] or "",
                confidence=float(row[self.confidence_col] or 0.5),
            )
        
        conn.close()


@dataclass  
class ScreenGhostDBSource(CaptureSource):
    """
    Read from a real ScreenGhost SQLite database.
    
    This is the bridge to your actual ScreenGhost runs.
    Schema expected:
    - table: captures
    - columns: id, timestamp, app_name, window_title, ocr_text, screenshot_path
    """
    
    db_path: Path
    topic_extractor: Optional[Callable[[str, str], str]] = None
    confidence: float = 0.7
    _last_id: int = 0
    
    def name(self) -> str:
        return f"screenghost:{self.db_path}"
    
    def events(self) -> Iterator[Event]:
        if not self.db_path.exists():
            return
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute("""
            SELECT id, timestamp, app_name, window_title, ocr_text
            FROM captures
            WHERE id > ?
            ORDER BY id ASC
        """, (self._last_id,))
        
        for row in cursor:
            self._last_id = row["id"]
            
            # Extract topic from app/window if extractor provided
            if self.topic_extractor:
                topic = self.topic_extractor(
                    row["app_name"] or "",
                    row["window_title"] or ""
                )
            else:
                topic = row["app_name"] or "screen_capture"
            
            yield Event(
                topic=topic,
                source=f"screenghost:{row['app_name']}",
                text=row["ocr_text"] or row["window_title"] or "",
                confidence=self.confidence,
            )
        
        conn.close()


# =============================================================================
# HTTP/Webhook Sources
# =============================================================================

@dataclass
class WebhookSource(CaptureSource):
    """
    Receive events via HTTP webhook.
    
    Starts a background server that accepts POST requests.
    Events are queued and yielded on demand.
    """
    
    host: str = "0.0.0.0"
    port: int = 9999
    source_name: str = "webhook"
    _queue: Queue = field(default_factory=Queue)
    _server_thread: Optional[Thread] = None
    
    def name(self) -> str:
        return f"webhook:{self.host}:{self.port}"
    
    def start(self) -> None:
        """Start the webhook server in background."""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        queue = self._queue
        source_name = self.source_name
        
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                
                try:
                    data = json.loads(body)
                    event = Event(
                        topic=data.get("topic", "webhook"),
                        source=data.get("source", source_name),
                        text=data.get("text", ""),
                        confidence=float(data.get("confidence", 0.5)),
                    )
                    queue.put(event)
                    self.send_response(200)
                except Exception:
                    self.send_response(400)
                
                self.end_headers()
            
            def log_message(self, format, *args):
                pass  # Suppress logging
        
        server = HTTPServer((self.host, self.port), Handler)
        self._server_thread = Thread(target=server.serve_forever, daemon=True)
        self._server_thread.start()
    
    def events(self) -> Iterator[Event]:
        while not self._queue.empty():
            yield self._queue.get_nowait()


@dataclass
class PollingSource(CaptureSource):
    """
    Poll an HTTP endpoint for events.
    
    Expected response format:
    {"events": [{"topic": "...", "source": "...", "text": "...", "confidence": 0.9}]}
    """
    
    url: str
    poll_interval: float = 5.0
    source_name: str = "poll"
    _last_poll: float = 0
    
    def name(self) -> str:
        return f"poll:{self.url}"
    
    def events(self) -> Iterator[Event]:
        import urllib.request
        
        now = time.time()
        if now - self._last_poll < self.poll_interval:
            return
        
        self._last_poll = now
        
        try:
            with urllib.request.urlopen(self.url, timeout=5) as resp:
                data = json.loads(resp.read())
                for item in data.get("events", []):
                    yield Event(
                        topic=item.get("topic", "unknown"),
                        source=item.get("source", self.source_name),
                        text=item.get("text", ""),
                        confidence=float(item.get("confidence", 0.5)),
                    )
        except Exception:
            pass  # Silently skip failed polls


# =============================================================================
# Multiplexer
# =============================================================================

@dataclass
class MultiplexSource(CaptureSource):
    """
    Combine multiple capture sources into one stream.
    
    This is how you connect multiple feeds to a single GhostBox instance.
    """
    
    sources: List[CaptureSource] = field(default_factory=list)
    
    def name(self) -> str:
        names = [s.name() for s in self.sources]
        return f"multiplex:[{', '.join(names)}]"
    
    def add(self, source: CaptureSource) -> None:
        self.sources.append(source)
    
    def events(self) -> Iterator[Event]:
        for source in self.sources:
            yield from source.events()


# =============================================================================
# Topic Extractors
# =============================================================================

def app_based_topic(app_name: str, window_title: str) -> str:
    """Extract topic from app name."""
    app_lower = app_name.lower()
    
    if "slack" in app_lower:
        return "slack_comms"
    elif "outlook" in app_lower or "mail" in app_lower:
        return "email"
    elif "chrome" in app_lower or "firefox" in app_lower or "safari" in app_lower:
        return "web_browsing"
    elif "terminal" in app_lower or "iterm" in app_lower:
        return "terminal"
    elif "code" in app_lower or "vim" in app_lower:
        return "coding"
    elif "calendar" in app_lower:
        return "calendar"
    else:
        return app_name.lower().replace(" ", "_")


def keyword_topic(keywords: Dict[str, List[str]]) -> Callable[[str, str], str]:
    """
    Create a topic extractor based on keyword matching.
    
    Usage:
        extractor = keyword_topic({
            "custody": ["custody", "visitation", "parenting"],
            "work": ["meeting", "project", "deadline"],
        })
    """
    def extract(app_name: str, window_title: str) -> str:
        text = f"{app_name} {window_title}".lower()
        for topic, words in keywords.items():
            if any(w in text for w in words):
                return topic
        return "general"
    
    return extract
