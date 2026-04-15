import json
import os
import threading
from datetime import datetime


class RuntimeLogger:
    """
    轻量级结构化日志（JSON Lines）:
    - 控制台输出简版，便于实时查看
    - 文件输出全量字段，便于离线分析
    """
    def __init__(self, log_dir="./logs"):
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.log_dir, f"runtime_{ts}.jsonl")
        self._lock = threading.Lock()

    def log(self, event, **fields):
        payload = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "event": event,
        }
        payload.update(fields)

        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

        msg = payload.get("message", "")
        if not msg and payload.get("text"):
            text = str(payload.get("text"))
            msg = text if len(text) <= 80 else f"{text[:77]}..."
        latency_ms = payload.get("latency_ms")
        if latency_ms is not None:
            print(f"[{payload['ts']}] {event} latency={latency_ms:.1f}ms {msg}")
        else:
            print(f"[{payload['ts']}] {event} {msg}")
