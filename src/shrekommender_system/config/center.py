"""Simple polling based configuration centre."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional

from .loader import load_yaml

Callback = Callable[[Dict], None]


class ConfigCenter:
    """Polls configuration files and notifies subscribers on change."""

    def __init__(self, config_dir: Path, poll_interval: float = 5.0):
        self.config_dir = config_dir
        self.poll_interval = poll_interval
        self._watchers: Dict[str, Callback] = {}
        self._mtimes: Dict[str, float] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def register(self, relative_path: str, callback: Callback) -> None:
        target = (self.config_dir / relative_path).resolve()
        if not target.exists():
            raise FileNotFoundError(f"Configuration file not found: {target}")
        self._watchers[relative_path] = callback
        self._mtimes[relative_path] = target.stat().st_mtime
        callback(load_yaml(target))

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.poll_interval * 2)

    def _loop(self) -> None:
        while not self._stop.is_set():
            for relative_path, callback in list(self._watchers.items()):
                path = (self.config_dir / relative_path).resolve()
                try:
                    current_mtime = path.stat().st_mtime
                except FileNotFoundError:
                    continue
                previous = self._mtimes.get(relative_path)
                if previous is None or current_mtime > previous:
                    self._mtimes[relative_path] = current_mtime
                    payload = load_yaml(path)
                    callback(payload)
            time.sleep(self.poll_interval)
