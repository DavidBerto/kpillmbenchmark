"""Cache persistente de respostas (SQLite). Torna reruns grátis e a rodada retomável."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from kpibench.llm.base import LLMResponse


def cache_key(model_id: str, api_model: str, params: dict, system: str, user: str) -> str:
    blob = json.dumps([model_id, api_model, params, system, user], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class ResponseCache:
    def __init__(self, path: Path | None):
        self.enabled = path is not None
        self._lock = threading.Lock()
        if self.enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
            self._conn.execute("CREATE TABLE IF NOT EXISTS responses (k TEXT PRIMARY KEY, v TEXT)")
            self._conn.commit()

    def get(self, key: str) -> LLMResponse | None:
        if not self.enabled:
            return None
        with self._lock:
            row = self._conn.execute("SELECT v FROM responses WHERE k=?", (key,)).fetchone()
        return LLMResponse.from_dict(json.loads(row[0])) if row else None

    def put(self, key: str, resp: LLMResponse) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO responses VALUES (?, ?)",
                               (key, json.dumps(resp.to_dict(), ensure_ascii=False)))
            self._conn.commit()
