from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class RunReporter:
    """Lightweight console and JSONL event reporting for one experiment."""

    def __init__(self, run_id: str, run_dir: Path) -> None:
        self.run_id = run_id
        self.path = run_dir / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **details: Any) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "event": event,
            **details,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        readable = " ".join(f"{key}={value}" for key, value in details.items())
        print(f"[{event}] {readable}".rstrip(), flush=True)

    @contextmanager
    def stage(self, name: str, **details: Any) -> Iterator[None]:
        started = time.perf_counter()
        self.emit("stage_started", stage=name, **details)
        try:
            yield
        except Exception as exc:
            self.emit(
                "stage_failed",
                stage=name,
                duration_seconds=round(time.perf_counter() - started, 3),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        self.emit(
            "stage_completed",
            stage=name,
            duration_seconds=round(time.perf_counter() - started, 3),
        )
